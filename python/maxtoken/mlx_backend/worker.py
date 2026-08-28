"""MLX scheduler worker: the Apple-silicon counterpart of ``scheduler.Scheduler``.

Protocol contract (mirrors the CUDA scheduler, see ``scheduler/io.py`` and
``tokenizer/server.py``):

- Pulls ``BaseBackendMsg`` from ``config.zmq_backend_addr`` (this side binds).
- Pushes ``BaseTokenizerMsg`` into ``config.zmq_detokenizer_addr``:
  ``PromptAdmittedMsg`` once per admitted prompt (usage accounting),
  ``DetokenizeMsg`` per sampled token (the detokenizer turns them into text),
  ``ErrorReplyMsg`` for requests that cannot be served,
  ``CacheRebuildResultMsg`` for runtime cache-rebuild control requests.

Execution is delegated to mlx-lm (one ``generate_step`` generator per request,
each owning its KV cache). Requests are stepped round-robin, one token each per
loop turn, so several concurrent requests all stream — no request starves behind
another's full completion. EOS, ``max_tokens`` and stop-string termination are
detected here (the CUDA scheduler's split of responsibilities): the detokenizer
worker only renders text and trims at ``matched_stop``.
"""

from __future__ import annotations

import inspect
import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

from maxtoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from maxtoken.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_eos_token_ids

from . import gdn_capture, spec_sample

# Longest verify window a speculative round will build.
#
# Four rows is the widest the small-M quantized matmul kernel compiles for
# (verify_qmm.py); past it MLX's own multi-row path takes over and the extra
# rows stop being cheap. A fifth row measured no better than the fourth even
# when it fit (23.20 against 23.27 tok/s), so four is both the ceiling and the
# right answer.
MAX_WINDOW = 4

if TYPE_CHECKING:
    from maxtoken.core import SamplingParams
    from maxtoken.scheduler import SchedulerConfig

logger = init_logger(__name__)


@dataclass
class _MlxRequest:
    uid: int
    sampling_params: SamplingParams
    generator: Iterator[Any]
    prompt_len: int
    output_ids: List[int] = field(default_factory=list)
    # For the prefix store: the exact prompt tokens and the live cache objects
    # (None when prefix caching is off). See MlxScheduler._remember.
    prompt_ids: List[int] = field(default_factory=list)
    cache: Any = None


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs ``fn`` accepts — mlx-lm's sampler/step signatures gain and
    lose knobs between minor versions, and an unexpected-kwarg crash on a version
    bump would take the whole backend down for parameters we can safely drop."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


class MlxScheduler:
    """Single-process MLX scheduler. Not thread-safe; owns the process's GPU state."""

    def __init__(self, config: SchedulerConfig):
        # The Metal command-buffer limits stay MLX's own unless the environment
        # sets them -- see metal_env.py for what happened when the worker chose
        # wide buffers itself. Logged so a benchmark states what it ran under.
        from .metal_env import dispatch_limits

        limits = {k: v for k, v in dispatch_limits().items() if v is not None}
        if limits:
            logger.info(f"Metal command-buffer limits from the environment: {limits}")
        import mlx.core as mx  # noqa: F401 -- fail here, before any socket binds
        from mlx_lm import load

        self._mx = mx
        _preimport_architectures()
        # mlx-lm resolves both local paths and hub ids (through the HF cache),
        # matching the tokenizer workers' resolution.
        logger.info(f"Loading MLX model from {config.model_path}")
        load = _wrap_fast_quantized(load)
        self.offload_state = None
        offload_wanted = (
            getattr(config, "moe_backend", "auto") in ("offload", "cpu", "hybrid")
            or getattr(config, "moe_cache_size", 0) > 0
            or getattr(config, "moe_cache_rate", None) is not None
            or getattr(config, "moe_cache_auto", False)
        ) and getattr(config, "moe_backend", "auto") != "fused"
        if offload_wanted:
            self.model, self.tokenizer = load(config.model_path, lazy=True)
            self._attach_offload(config)
        else:
            self.model, self.tokenizer = load(config.model_path)
        self.draft = None
        self._spec_steps = 0
        self._spec_tokens = 0
        if getattr(config, "draft_model", None):
            self._attach_draft(config)
        # Resident and mapped serving: fuse the gated-delta input projections
        # and compile the MoE blocks for the decode shape (decode_fusion.py).
        # After the mapped-expert surgery, since compiled blocks bind the arrays
        # they were traced with; never with a slot cache, which routes experts
        # itself. MAXTOKEN_MLX_DECODE_FUSION=0 keeps the stock forward.
        import os as _os

        if self.offload_state is None and _os.environ.get(
            "MAXTOKEN_MLX_DECODE_FUSION", "1"
        ) != "0":
            from . import decode_fusion

            decode_fusion.install(self.model)
        # Continuous batching (resident and mapped-expert serving): concurrent
        # requests decode in ONE batched forward per step instead of one forward
        # per request per token. The slot-cache offload path keeps its own
        # speculate/verify loop and stays round-robin.
        self.batch_gen = None
        self._batch_uid: dict[int, int] = {}  # our uid -> engine uid
        self._our_uid: dict[int, int] = {}  # engine uid -> our uid
        # Kept beside the geometry rather than read from self.config: the batcher
        # is built here, and self.config is not assigned until further down.
        self._batch_running = max(1, config.max_running_req)
        if self.offload_state is None and self.draft is None:
            self._prefill_batch, self._prefill_step = self._prefill_geometry(
                self._batch_running
            )
            self.batch_gen = self._make_batch_generator()
            logger.info(
                f"continuous batching: up to {max(1, config.max_running_req)} "
                f"concurrent decodes per forward, prefill {self._prefill_batch}"
                f"x{self._prefill_step} tokens"
            )
        hf_tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.eos_token_ids = frozenset(load_eos_token_ids(config.model_path, hf_tokenizer))
        self.config = config
        self.max_seq_len = int(config.max_seq_len)
        self._decode_steps = 0
        self.prefix_store = None
        if getattr(config, "cache_type", "radix") != "naive":
            import os as _os

            from .prefix_cache import PrefixStore

            budget = int(_os.environ.get("MAXTOKEN_MLX_PREFIX_CACHE_MB", "0")) * 2**20
            if budget <= 0:
                try:
                    budget = int(0.15 * mx.metal.device_info()["memory_size"])
                except Exception:  # noqa: BLE001 -- conservative fallback
                    budget = 2 << 30
            self.prefix_store = PrefixStore(budget)
            logger.info(
                f"prefix cache: on ({budget / 2**30:.1f} GiB budget; "
                "--cache-type naive disables)"
            )

        self._recv = ZmqPullQueue(
            config.zmq_backend_addr, create=True, decoder=BaseBackendMsg.decoder
        )
        self._send = ZmqPushQueue(
            config.zmq_detokenizer_addr,
            create=config.backend_create_detokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        )
        self.active: Dict[int, _MlxRequest] = {}

    # ------------------------------------------------------------------ offload

    def _attach_offload(self, config: SchedulerConfig) -> None:
        """Wire the expert slot cache in (MaxToken's core: serve a model whose
        experts don't fit the memory budget). Falls back to fully resident when
        the checkpoint has no offloadable experts (dense model)."""
        import os

        from .offload import attach_expert_offload

        mx = self._mx
        if os.path.isdir(config.model_path):
            model_dir = config.model_path
        else:
            # Same resolution mlx_lm.load uses (weights are already cached by it);
            # restricted patterns so a snapshot without README/.gitattributes
            # still validates offline.
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                config.model_path,
                local_files_only=True,
                allow_patterns=["*.safetensors", "*.json"],
            )
        explicit_budget = (
            getattr(config, "moe_cache_size", 0) > 0
            or getattr(config, "moe_cache_rate", None)
        )
        if not explicit_budget:
            # Default offload on MLX: the zero-copy mmapped expert store. Serving
            # runs at resident-kernel speed with OS-managed residency (page
            # cache); no slot cache, no budget knob. An explicit --moe-cache-*
            # size selects the slot-cache path below with its hard budget.
            from .ftw_mlx import attach_mapped_experts

            try:
                attach_mapped_experts(self.model, model_dir)
            except ValueError as exc:
                logger.warning(
                    f"mapped expert store unavailable ({exc}); serving fully resident"
                )
            mx.eval(self.model.parameters())
            logger.info(
                f"dense+mapped resident: {mx.get_active_memory() / 2**30:.2f} GiB "
                "active (expert pages are file-backed page cache on top)"
            )
            return
        try:
            # Attach with minimal slots first: auto-sizing needs the resident
            # (dense-weights) footprint, which exists only after the surgery
            # dropped the expert stacks and the rest got evaluated.
            self.offload_state = attach_expert_offload(self.model, model_dir, 1)
        except ValueError as exc:
            logger.warning(f"expert offload unavailable ({exc}); serving fully resident")
            mx.eval(self.model.parameters())
            return
        mx.eval(self.model.parameters())
        dense_bytes = mx.get_active_memory()
        per_layer = self._resolve_slots_per_layer(config, dense_bytes)
        total = self.offload_state.resize_total(per_layer * len(self.offload_state.glus))
        st = self.offload_state
        logger.info(
            f"MoE expert cache: {total} slots "
            f"({st.cache_bytes() / 2**30:.2f} GiB), dense-resident "
            f"{dense_bytes / 2**30:.2f} GiB"
        )

    def _attach_draft(self, config: SchedulerConfig) -> None:
        """Load the draft model for speculative decoding (--draft-model)."""
        from .draft import DraftModel

        k = max(1, int(getattr(config, "draft_tokens", 3) or 3))
        k = self._clamp_draft_k(k)
        if str(config.draft_model).strip().lower() == "mtp":
            # The checkpoint's own MTP head: one layer, sharing the trunk's
            # tokenizer and lm_head. See mtp_draft.py for why that shape wins.
            from .mtp_draft import MtpDrafter

            self.draft = MtpDrafter.load(self.model, self._model_dir(config), k)
        else:
            from .draft import _vocab_size

            self.draft = DraftModel.load(
                config.draft_model, k, target_vocab=_vocab_size(self.model)
            )
        if self.offload_state is not None:
            self.offload_state.spec_window = k + 1
        # A verify window is the one place a quantized matmul runs 2-4 rows,
        # and MLX's kernel charges nearly a full row's work for each of them.
        # Installing our own here — and only here, when a drafter exists — is
        # what makes the window cheap enough for speculation to pay at all;
        # without it the arithmetic loses before the drafter has done anything
        # (docs/mlx.md). Nothing else in the process changes: single-row decode
        # and prefill never enter the patched path.
        from . import verify_qmm

        verify_qmm.install()
        logger.info(
            f"speculative decoding: draft model {config.draft_model}, "
            f"k={k} tokens per verify"
        )

    @staticmethod
    def _model_dir(config: SchedulerConfig) -> str:
        """Local directory of the checkpoint (hub ids resolve through the cache
        mlx_lm.load already populated)."""
        import os

        if os.path.isdir(config.model_path):
            return config.model_path
        from huggingface_hub import snapshot_download

        return snapshot_download(
            config.model_path,
            local_files_only=True,
            allow_patterns=["*.safetensors", "*.json"],
        )

    def _clamp_draft_k(self, k: int) -> int:
        """The verify window's routed experts must fit each layer's slot cache,
        or the redo loop can never converge: (k+1) * top_k <= min slots. Without
        a slot cache (resident or mapped serving) nothing bounds the window."""
        st = self.offload_state
        if st is None:
            return k
        top_k = int(
            getattr(getattr(self.model, "args", None), "num_experts_per_tok", 0) or 8
        )
        min_slots = min(g.cache.num_slots for g in st.glus)
        max_window = max(2, min_slots // top_k)
        if k + 1 > max_window:
            logger.warning(
                f"--draft-tokens {k} clamped to {max_window - 1}: the expert "
                f"cache has {min_slots} slots/layer and top-{top_k} routing "
                f"needs the whole verify window resident at once"
            )
            k = max_window - 1
        return k

    def _resolve_slots_per_layer(self, config: SchedulerConfig, dense_bytes: int) -> int:
        st = self.offload_state
        n_layers = len(st.glus)
        num_experts = st.glus[0].store.num_experts
        per_expert = st.glus[0].store.expert_nbytes
        if getattr(config, "moe_cache_size", 0) > 0:
            per_layer = config.moe_cache_size // n_layers
        elif getattr(config, "moe_cache_rate", None):
            per_layer = int(config.moe_cache_rate * num_experts)
        else:
            # Auto: fill the memory budget with experts, like the CUDA engine's
            # --moe-cache-auto (KV gets a flat reserve; MLX KV is per-request).
            try:
                total_mem = self._mx.metal.device_info()["memory_size"]
            except Exception:  # noqa: BLE001 -- conservative fallback
                total_mem = 16 * 2**30
            budget = int(config.memory_ratio * total_mem) - dense_bytes - 2 * 2**30
            per_layer = max(1, budget // per_expert // n_layers)
        return max(16, min(num_experts, per_layer))

    # ------------------------------------------------------------------ generation

    def _build_sampler(self, sp: SamplingParams):
        from mlx_lm.sample_utils import make_sampler

        # Greedy needs no sampler even when top_p/top_k defaults are filled in
        # (they can never remove the argmax token). Returning None matters for
        # batching: any per-row sampler forces the engine off the batched
        # argmax fast path into a per-sequence sampling loop.
        if sp.is_greedy or sp.temperature <= 0.0 or sp.top_k == 1:
            return None
        # With a bounded top_k the sampling happens over that support alone
        # (spec_sample.device_sampler): half the cost of mlx-lm's sampler on a
        # 248k vocabulary, which sorts all of it for top-p. Shapes the
        # distribution the way the speculative loop does -- temperature, then
        # top-k, then top-p -- where mlx-lm applies top-p before temperature.
        spec = spec_sample.sampler_spec(sp)
        if spec is not None:
            return spec_sample.device_sampler(spec)
        return make_sampler(
            **_filter_kwargs(
                make_sampler,
                {
                    "temp": max(sp.temperature, 0.0),
                    # mlx-lm encodes "disabled" as 0 (maxtoken: 1.0 / -1).
                    "top_p": sp.top_p if 0.0 < sp.top_p < 1.0 else 0.0,
                    "top_k": sp.top_k if sp.top_k > 0 else 0,
                },
            )
        )

    def _lookup_prefix(self, input_ids: List[int]) -> tuple:
        """(restored cache | None, cached_tokens) from the prefix store."""
        if self.prefix_store is None:
            return None, 0
        hit = self.prefix_store.lookup(input_ids)
        if hit is None:
            return None, 0
        entry, n = hit
        return self.prefix_store.restore(self.model, entry, n), n

    def _prefill_into(
        self, cache, input_ids: List[int], start: int, on_hidden=None
    ) -> None:
        """Process input_ids[start:-1] into ``cache`` in chunks, snapshotting at
        BOUNDARY_TOKENS multiples so hybrid models (whose recurrent state cannot
        be trimmed) have exact restore points for future prefix hits.

        ``on_hidden(hidden, chunk_start, chunk_end)`` receives the trunk's hidden
        states for each chunk. Asking for them also drops the lm_head from the
        prefill: the prompt's logits are never read, and at this vocabulary a
        2048-token chunk would materialize a gigabyte of them.
        """
        from .prefix_cache import BOUNDARY_TOKENS

        mx = self._mx
        text = getattr(self.model, "language_model", self.model)
        pos = start
        end = len(input_ids) - 1
        next_boundary = (pos // BOUNDARY_TOKENS + 1) * BOUNDARY_TOKENS
        while pos < end:
            n = min(2048, end - pos, next_boundary - pos)
            chunk = mx.array(input_ids[pos:pos + n])[None]
            if on_hidden is None:
                mx.eval(self.model(chunk, cache=cache))
            else:
                hidden = text.model(chunk, cache=cache)
                mx.eval(hidden)
                on_hidden(hidden, pos, pos + n)
            pos += n
            if pos == next_boundary:
                # The boundary advances whether or not anything is stored at it.
                # Tying the advance to the prefix store left next_boundary
                # pinned as soon as the store was off (--cache-type naive), so
                # the following chunk clamped to zero tokens and the model was
                # handed an empty array — every prompt past BOUNDARY_TOKENS.
                if pos < end and self.prefix_store is not None:
                    self.prefix_store.insert(input_ids[:pos], cache)
                next_boundary += BOUNDARY_TOKENS

    def _make_generator(self, input_ids: List[int], sp: SamplingParams) -> tuple:
        """(token generator, live cache list | None, cached prefix tokens)."""
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        mx = self._mx
        cache, cached = self._lookup_prefix(input_ids)
        if self.draft is not None:
            cache = cache or make_prompt_cache(self.model)
            return self._generate_spec(input_ids, sp, cache, cached), cache, cached
        if self.offload_state is not None:
            cache = cache or make_prompt_cache(self.model)
            return self._offload_generate(input_ids, sp, cache, cached), cache, cached
        cache = cache or make_prompt_cache(self.model)
        self._prefill_into(cache, input_ids, cached)
        # max_tokens=-1 -> unbounded; EOS/length/stop are all enforced in _step so
        # ignore_eos and the exact CUDA-scheduler semantics stay in one place.
        kwargs = _filter_kwargs(
            generate_step,
            {"max_tokens": -1, "sampler": self._build_sampler(sp), "prompt_cache": cache},
        )
        gen = generate_step(mx.array(input_ids[-1:]), self.model, **kwargs)
        return gen, cache, cached

    @staticmethod
    def _cache_snapshot(c) -> Any:
        """Cheap per-step rollback point for one mlx-lm cache object.

        A plain KV cache rewinds by trimming, and that is the path worth
        keeping: holding no reference to its arrays lets the engine keep
        updating its buffers in place (a snapshot would force a copy-on-write
        of the whole cache every step).

        Everything else is restored from its arrays PLUS its ``meta_state``:
        recurrent GDN caches (not trimmable at all), and window caches, whose
        position lives in the meta state — ``RotatingKVCache.meta_state`` is
        ``(keep, max_size, offset, _idx)``. Restoring arrays alone leaves the
        offset past the buffer it just rewound and the next update computes a
        negative size (seen on DeepSeek-V4's sliding-window attention). Window
        caches are also only *conditionally* trimmable (``offset < max_size``),
        so a step that wraps the window would invalidate a trim planned before
        it — another reason they take the full-snapshot path.
        """
        if c.is_trimmable() and not hasattr(c, "max_size"):
            return None
        return (list(c.state), c.meta_state)

    @staticmethod
    def _cache_rollback(c, snap, n: int = 1) -> None:
        if snap is None:
            c.trim(n)
        else:
            state, meta = snap
            # A copy, never the snapshot's own list: ArraysCache keeps the list
            # it is handed and writes the next step's arrays INTO it, so a
            # snapshot restored twice would restore the state of the forward
            # that followed the first restore.
            c.state = list(state)
            c.meta_state = meta

    def _offload_prefill(self, input_ids: List[int], cache, start: int) -> None:
        """Prefill all tokens but the last: per-layer sync/streamed serving,
        with prefix-store snapshots at restore-safe boundaries."""
        from .prefix_cache import BOUNDARY_TOKENS

        mx = self._mx
        state = self.offload_state
        state.speculating = False
        pos, end = start, len(input_ids) - 1
        next_boundary = (pos // BOUNDARY_TOKENS + 1) * BOUNDARY_TOKENS
        while pos < end:
            n = min(2048, end - pos, next_boundary - pos)
            state.begin_token()
            logits = self.model(mx.array(input_ids[pos:pos + n])[None], cache=cache)
            mx.eval(logits)
            pos += n
            if pos == next_boundary:
                # The boundary advances whether or not anything is stored at it.
                # Tying the advance to the prefix store left next_boundary
                # pinned as soon as the store was off (--cache-type naive), so
                # the following chunk clamped to zero tokens and the model was
                # handed an empty array — every prompt past BOUNDARY_TOKENS.
                if pos < end and self.prefix_store is not None:
                    self.prefix_store.insert(input_ids[:pos], cache)
                next_boundary += BOUNDARY_TOKENS

    def _shaped_dist(self, sp: SamplingParams):
        """A function logits-row -> the probability vector the request actually
        samples from (temperature, then top-k, then top-p, renormalized), or
        None for greedy. Speculative acceptance needs this EXPLICITLY: the
        guarantee is that committed tokens follow exactly this distribution."""
        mx = self._mx
        if sp.is_greedy or sp.temperature <= 0.0:
            return None
        temp = max(float(sp.temperature), 1e-6)
        top_k = int(sp.top_k) if sp.top_k and sp.top_k > 0 else 0
        top_p = float(sp.top_p) if 0.0 < sp.top_p < 1.0 else 0.0

        def shape(row):
            probs = mx.softmax(row.astype(mx.float32) / temp, axis=-1)
            if top_k:
                kth = mx.argpartition(probs, -top_k)[-top_k:]
                keep = mx.zeros_like(probs)
                keep[kth] = 1.0
                probs = probs * keep
            if top_p:
                # Descending order via argsort(-probs), NOT argsort(probs)[::-1]:
                # scatter-assigning through a reversed (negative-stride) index
                # view writes only its first element on MLX 0.32, which
                # silently collapsed every top-p distribution to its argmax —
                # the "sampled" path was greedy in disguise (see
                # test_spec_sample.py::test_dense_shape_matches_numpy).
                order = mx.argsort(-probs)
                ordered = probs[order]
                # keep the smallest prefix whose mass reaches top_p
                cum = mx.cumsum(ordered)
                keep_sorted = (cum - ordered) < top_p
                keep = mx.zeros_like(probs)
                keep[order] = keep_sorted.astype(probs.dtype)
                probs = probs * keep
            total = probs.sum()
            return probs / mx.maximum(total, 1e-30)

        return shape

    def _accept_speculative(self, drafts, P, Q, rng_key=None):
        """Rejection sampling with residual correction (Leviathan et al.).

        Accept draft ``d`` at position i with probability ``min(1, p(d)/q(d))``;
        on the first rejection draw from the normalized residual ``(p - q)+``
        and stop. Committed tokens are distributed exactly as ``p`` — the
        target's own sampling distribution — while accepting strictly more
        drafts than sample-and-match whenever the drafter is less confident
        than the target. Returns (accepted_count, token_after_the_prefix).
        """
        mx = self._mx
        for i, d in enumerate(drafts):
            p_d = float(P[i][d].item())
            q_d = float(Q[i][d].item())
            if q_d <= 0.0:
                ratio = 0.0
            else:
                ratio = min(1.0, p_d / q_d)
            if float(mx.random.uniform().item()) < ratio:
                continue
            residual = mx.maximum(P[i] - Q[i], 0.0)
            total = float(residual.sum().item())
            dist = P[i] if total <= 0.0 else residual / total
            return i, int(mx.random.categorical(mx.log(dist + 1e-30)).item())
        # every draft survived: the bonus token comes from the last position
        last = P[len(drafts)]
        return len(drafts), int(mx.random.categorical(mx.log(last + 1e-30)).item())

    def _forward(self, tokens, cache, want_hidden: bool):
        """(logits, hidden | None). mlx-lm models return logits only; the hidden
        state comes from running the text tower and its head separately, which
        an MTP drafter consumes."""
        if not want_hidden:
            return self.model(tokens, cache=cache), None
        text = getattr(self.model, "language_model", self.model)
        hidden = text.model(tokens, cache=cache)
        head = getattr(text, "lm_head", None)
        logits = head(hidden) if head is not None else text.model.embed_tokens.as_linear(hidden)
        return logits, hidden

    def _generate_spec(
        self, input_ids: List[int], sp: SamplingParams, cache, start: int = 0
    ) -> Iterator[Any]:
        """Speculative decode with a draft model, for every serving mode.

        The draft model proposes k tokens, the target verifies the whole window
        in ONE forward and commits the matched prefix plus one target token
        (correction or bonus). Distribution-exact: every emitted token is the
        target's own argmax/sample for its position; drafts only decide how many
        positions a single forward advances.

        Where it pays differs by mode. Resident and mapped serving are
        BANDWIDTH-bound — a decode step pushes all active weights through
        memory — so amortizing one forward over several tokens is the whole
        point. Slot-cache offload is FETCH-bound instead: its cost scales with
        the tokens generated, not the forwards taken, so speculation there
        measured at parity (documented in docs/mlx.md).

        With a slot cache the verify runs the per-layer SYNC path (misses
        installed inline), not the lazy lut path: a missed expert garbles every
        layer downstream of it, so lut-mode redos cascade one wave of misses per
        attempt (measured 4.3 redo forwards per verify on Qwen3-Next-80B) while
        the sync path pays each layer exactly once. Rejected drafts leave the
        caches advanced over tokens that never happened — trimmable caches trim,
        recurrent and window ones restore the pre-window snapshot, and the
        accepted prefix is replayed (one extra forward, only on partial
        accepts).
        """
        mx = self._mx
        state = self.offload_state
        drafter = self.draft
        sampler = self._build_sampler(sp)
        k = drafter.k
        # Committed tokens the caches have not absorbed yet. A rejected round
        # leaves them here instead of paying a separate forward to replay them:
        # carried into the NEXT window they cost one extra row (~14 ms with the
        # verify kernel) instead of a whole forward (~56 ms). MAX_WINDOW caps
        # the carry, which is what keeps it from growing without bound — a
        # round that cannot fit any draft still forwards its pending tokens and
        # so always advances the caches.
        max_window = max(k + 1, MAX_WINDOW)
        # An MTP drafter reads the trunk's hidden state; a second model does not.
        # It gets its FIRST such state from the first window's own forward rather
        # than from a priming pass: priming ran the last prompt token through the
        # trunk, and the first window then fed it a SECOND time, leaving the
        # token duplicated in the KV sequence for the rest of the request. The
        # cost of waiting is one round without drafts, which is cheaper than the
        # priming forward it replaces.
        wants_hidden = hasattr(drafter, "set_hidden")
        drafter.start(input_ids)
        # The head's history is built from the same prefill that fills the
        # trunk's cache, so the drafter enters the request already conditioned on
        # the prompt. A prefix-cache hit skips the tokens it restored, and the
        # head simply starts its history later — a shorter history, never a
        # wrong one.
        history_sink = None
        if wants_hidden and hasattr(drafter, "extend_history"):
            def history_sink(hidden, chunk_start, chunk_end):
                drafter.extend_history(
                    hidden, input_ids[chunk_start + 1 : chunk_end + 1]
                )

        if state is not None:
            self._offload_prefill(input_ids, cache, start)
            state.spec_window = max_window
        else:
            self._prefill_into(cache, input_ids, start, on_hidden=history_sink)
        pending = [int(input_ids[-1])]
        eos = self.eos_token_ids

        # Sampled requests use rejection sampling with residual correction; that
        # needs the proposal density q, so the drafter must SAMPLE rather than
        # take its argmax (a deterministic proposal makes min(1, p/q) collapse
        # to p(d), i.e. no better than plain sample-and-match). Greedy requests
        # keep the exact-match rule, which is optimal when p is a point mass.
        shape = self._shaped_dist(sp)
        # A drafter that builds its chain lazily lets the round run with ONE
        # synchronization: the draft tokens stay on the device and feed the
        # window directly, the GPU starts drafting while the window's graph is
        # still being built on the CPU, and the target's distributions come back
        # as their top-k supports in a single transfer (spec_sample). A sampler
        # without a bounded support keeps the dense path.
        spec = spec_sample.sampler_spec(sp)
        lazy_round = hasattr(drafter, "draft_lazy") and (spec is not None or shape is None)
        rng = spec_sample.host_rng(mx) if lazy_round and spec is not None else None
        can_reject_sample = not lazy_round and shape is not None and hasattr(drafter, "q")
        # Recording per-position recurrent state lets a rejected round commit
        # its accepted prefix outright, which is what keeps the carry at a
        # single token and leaves the next window its full draft depth. Models
        # whose recurrent layers this cannot record keep the snapshot path.
        can_commit_prefix = self.offload_state is None and gdn_capture.supported(
            self.model
        )

        trace = getattr(self, "_spec_trace", None)
        clock = _time.perf_counter if trace is not None else None

        def draft_next(room: int):
            """Build the next round's draft chain and dispatch it. Called BEFORE
            the round's tokens are handed out, so the GPU drafts while the
            scheduler is busy with them and the CPU then builds the window
            graph on top of a busy GPU."""
            lz = drafter.draft_lazy(room, spec)
            if lz.n:
                mx.async_eval(lz.tokens, *lz.q)
            return lz

        # Drafts only fill what the window has left after the carried tokens.
        # When nothing is left the round runs the carry alone: it still commits
        # a token and, crucially, it absorbs the carry, so `pending` can never
        # outgrow the window.
        lazy = draft_next(max_window - len(pending)) if lazy_round else None
        t_resume = clock() if clock else 0.0
        while True:
            room = max_window - len(pending)
            if lazy_round:
                drafts_arr, n_drafts, drafts = lazy.tokens, lazy.n, None
            else:
                drafts = []
                if room > 0:
                    drafts = (
                        drafter.draft(shape) if can_reject_sample else drafter.draft()
                    ) or []
                    drafts = drafts[:room]
                drafts_arr, n_drafts = mx.array(drafts, dtype=mx.int32), len(drafts)
            window_len = len(pending) + n_drafts
            window_arr = mx.array(pending, dtype=mx.int32)
            if n_drafts:
                window_arr = mx.concatenate([window_arr, drafts_arr])
            # Draft j is judged by the target's output at window position
            # off + j; with a single carried token off is 0, which is the plain
            # speculative case.
            off = len(pending) - 1
            if state is not None:
                state.prefetch_predicted()
                state.speculating = False
                state.begin_token()
            # Taken even when the capture path is expected to commit: it costs
            # 0.02 ms a round and it is the only way back if commit_prefix
            # declines the window for a reason `supported` cannot see.
            snaps = [self._cache_snapshot(c) for c in cache]
            with gdn_capture.capture(self.model if can_commit_prefix else None) as caps:
                logits, hidden = self._forward(window_arr[None], cache, wants_hidden)
            # Nothing is normalized or sampled here. Each window position carries
            # a vocabulary-wide row (248k floats on this checkpoint), and the two
            # acceptance rules below need different things from them: rejection
            # sampling builds its own shaped distribution, so normalizing the
            # whole window first computed the target distribution twice per round
            # and threw one away.
            if state is not None:
                state.commit_token()
            t_built = clock() if clock else 0.0

            if lazy_round and spec is not None:
                # The round's one synchronization: draft tokens, the drafter's
                # proposal densities and the target's supports, together.
                p_ids, p_probs = spec_sample.support(
                    logits[0, off : off + n_drafts + 1], spec
                )
                mx.eval(drafts_arr, p_ids, p_probs, *lazy.q)
                t_done = clock() if clock else 0.0
                drafts = drafts_arr.tolist()
                P = spec_sample.shape_rows(p_ids, p_probs, spec)
                Q = spec_sample.rows_of(*lazy.q) if n_drafts else []
                accepted, nxt = spec_sample.accept(drafts, P, Q, rng)
                committed = drafts[:accepted] + [nxt]
            elif can_reject_sample and drafts and len(drafter.q) >= len(drafts):
                P = [shape(logits[0, off + i]) for i in range(len(drafts) + 1)]
                accepted, nxt = self._accept_speculative(
                    drafts, P, drafter.q[: len(drafts)]
                )
                t_done = clock() if clock else 0.0
                committed = drafts[:accepted] + [nxt]
            else:
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                outs = sampler(lp[0]) if sampler else mx.argmax(lp[0], axis=-1)
                mx.eval(outs, drafts_arr)
                t_done = clock() if clock else 0.0
                if drafts is None:
                    drafts = drafts_arr.tolist()
                outs_l = [int(t) for t in outs.tolist()]
                accepted = 0
                while accepted < len(drafts) and drafts[accepted] == outs_l[off + accepted]:
                    accepted += 1
                committed = outs_l[off : off + accepted + 1]
            # Stop the window at EOS: everything after it would over-advance the
            # caches past what the scheduler will ever consume (ignore_eos
            # requests keep the full window and let the scheduler decide).
            if not sp.ignore_eos:
                for i, tok in enumerate(committed):
                    if tok in eos:
                        committed, accepted = committed[: i + 1], min(accepted, i)
                        break

            # The drafter keeps its own history of what was committed; row
            # off + i of the window is the state that produced committed[i], so
            # the head is re-conditioned on the TARGET's hidden states rather
            # than on its own chained ones.
            if wants_hidden and hidden is not None and hasattr(drafter, "absorb"):
                drafter.absorb(committed, hidden[:, off : off + len(committed), :])

            if accepted < len(drafts):
                # Rejected drafts advanced the caches over tokens that never
                # happened. With per-position state recorded, the caches are
                # simply bound to the accepted prefix and the round ends like
                # any other — nothing recomputed, nothing carried.
                keep = off + accepted + 1
                committed_prefix = bool(caps) and gdn_capture.commit_prefix(
                    self.model, cache, caps, keep, window_len
                )
                if committed_prefix:
                    # The caches now hold `pending + drafts[:accepted]`. The only
                    # emitted token they have not absorbed is the correction.
                    pending = committed[-1:]
                else:
                    # No capture: roll back to the pre-window state and carry the
                    # committed path into the next window instead of replaying it
                    # in a forward of its own. The caches lose a round of
                    # progress, but the next window reabsorbs it at the price of
                    # extra rows rather than an extra forward.
                    for c, snap in zip(cache, snaps, strict=True):
                        self._cache_rollback(c, snap, window_len)
                    pending = pending + drafts[:accepted] + committed[-1:]
                drafter.commit(accepted, committed[-1:])
            else:
                # Every draft survived: the window IS the committed path and the
                # caches already hold it. Only the bonus token is left over.
                pending = committed[-1:]
                drafter.commit(accepted, (drafts[-1:] + committed[-1:]) if drafts
                               else committed[-1:])
            if wants_hidden and hidden is not None:
                # The state that PRODUCED the last committed token sits at
                # window index off + accepted — not at the window's end.
                # Feeding the last one drifts the head off the committed path
                # (measured: acceptance 0.25 vs 1.10 out of 3).
                j = off + accepted
                drafter.set_hidden(hidden[:, j : j + 1, :])
            t_post = clock() if clock else 0.0

            self._spec_steps += 1
            self._spec_tokens += len(committed)
            # The next chain is dispatched before this round's tokens leave the
            # generator: the scheduler's per-token work (and, in the server, a
            # reply round trip per token) then overlaps the drafter's GPU time
            # instead of adding to the round.
            if lazy_round:
                lazy = draft_next(max_window - len(pending))
            if trace is not None:
                t_drafted = clock()
                trace.append({
                    "away": 0.0,
                    "build": t_built - t_resume, "wait": t_done - t_built,
                    "post": t_post - t_done, "draft": t_drafted - t_post,
                    "tokens": len(committed), "accepted": accepted,
                })
            for i, tok in enumerate(committed):
                # Normalize only the row being emitted: the scheduler wants one
                # logprob row per committed token, which is a subset of the
                # window's positions.
                row = logits[:, off + i, :]
                yield tok, row - mx.logsumexp(row, axis=-1, keepdims=True)
            if trace is not None:
                t_now = clock()
                trace[-1]["away"] = t_now - t_drafted
                t_resume = t_now

    def _offload_generate(
        self, input_ids: List[int], sp: SamplingParams, cache, start: int = 0
    ) -> Iterator[Any]:
        """Decode loop for expert-offload serving: speculate-and-verify.

        Each step runs fully lazily against the device-side slot LUT (zero CPU
        syncs). One eval per token also brings back the per-layer "were all routed
        experts resident" flags; a miss rolls the KV/recurrent caches back one
        step, installs the missing experts and re-runs — so misses cost one extra
        forward, and the steady state runs at resident-model speed.
        """
        mx = self._mx
        state = self.offload_state
        sampler = self._build_sampler(sp)
        self._offload_prefill(input_ids, cache, start)
        y = mx.array(input_ids[-1:])

        # Adaptive serving: speculation wins when redos are rare (cache covers the
        # decode working set); per-layer sync serving wins when they are not.
        # Track a redo EMA, switch on a threshold, and probe speculation again
        # after a sync streak so a phase change (topic shift ends, cache warmed)
        # is noticed.
        redo_ema = 0.0
        sync_streak = 0
        while True:
            state.prefetch_predicted()  # last token's routing predicts this one's
            # Speculation wins whenever a redo round-trip (fast zero-sync forward)
            # beats 40 per-layer syncs — in practice almost always; the sync mode
            # remains as an escape hatch for pathological non-convergence.
            speculate = redo_ema < 0.9 or sync_streak >= 32
            state.speculating = speculate
            if speculate:
                sync_streak = 0
                redos = 0
                for _attempt in range(len(state.glus) + 1):
                    state.begin_token()
                    snaps = [self._cache_snapshot(c) for c in cache]
                    logits = self.model(y[None], cache=cache)
                    logprobs = logits[:, -1, :] - mx.logsumexp(
                        logits[:, -1, :], keepdims=True
                    )
                    y_next = sampler(logprobs) if sampler else mx.argmax(logprobs, axis=-1)
                    mx.eval(y_next, *state.pending_oks())
                    if state.commit_token():
                        break
                    redos += 1
                    for c, snap in zip(cache, snaps, strict=True):
                        self._cache_rollback(c, snap)
                else:  # pragma: no cover -- each round installs at least one layer
                    raise RuntimeError("expert cache failed to converge on a decode step")
                # EMA over EXTRA forwards per token; > ~2 sustained means sync wins.
                redo_ema = 0.85 * redo_ema + 0.15 * min(redos / 2.5, 1.0)
            else:
                sync_streak += 1
                state.begin_token()  # sync path installs inline; nothing pends
                logits = self.model(y[None], cache=cache)
                logprobs = logits[:, -1, :] - mx.logsumexp(
                    logits[:, -1, :], keepdims=True
                )
                y_next = sampler(logprobs) if sampler else mx.argmax(logprobs, axis=-1)
                mx.eval(y_next)
            yield int(y_next.item()), logprobs
            y = y_next

    def _maybe_rebalance(self) -> None:
        """Between requests: redistribute the slot budget by observed per-layer
        miss pressure (MAXTOKEN_MLX_REBALANCE=0 disables). Idle-only — resize
        rebuilds the device lut/owner arrays and must not race a forward."""
        import os as _os

        if self.offload_state is None:
            return
        if _os.environ.get("MAXTOKEN_MLX_REBALANCE", "1") == "0":
            return
        floor = 16
        if self.draft is not None:
            top_k = int(
                getattr(getattr(self.model, "args", None), "num_experts_per_tok", 0)
                or 8
            )
            floor = max(floor, self.offload_state.spec_window * top_k)
        if self.offload_state.rebalance(floor=floor) and self.draft is not None:
            self.draft.k = self._clamp_draft_k(self.draft.k)
            self.offload_state.spec_window = self.draft.k + 1

    def _match_stop_str(self, req: _MlxRequest) -> str | None:
        """First stop string in the generated tail, else None. Same bound as the CUDA
        scheduler's ``_match_stop_str``: a stop of N chars spans at most N tokens."""
        stop_strs = req.sampling_params.stop_strs
        if not stop_strs or not req.output_ids:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail = self.tokenizer.decode(req.output_ids[-(max_chars + 1):])
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _step(self) -> None:
        """Advance every active request by one token and ship the replies."""
        reply: List[BaseTokenizerMsg] = []
        gpu_mem = int(self._mx.get_active_memory())
        if self.offload_state is not None:
            self._decode_steps += 1
            if self._decode_steps % self.config.decode_log_interval == 0:
                h, m, slots = self.offload_state.totals()
                rate = m / max(1, h + m)
                spec = ""
                if self._spec_steps:
                    spec = (
                        f", speculative accept {self._spec_tokens / self._spec_steps:.2f}"
                        f" tok/verify ({self._spec_tokens}/{self._spec_steps})"
                    )
                logger.info(
                    f"expert cache: {slots} slots, lifetime miss rate {rate:.1%} "
                    f"({m}/{h + m}), active mem {gpu_mem / 2**30:.2f} GiB{spec}"
                )
        if self.batch_gen is not None:
            try:
                self._step_batched(reply, gpu_mem)
            except Exception as exc:  # noqa: BLE001 -- isolate: the batch, not the worker
                # The round-robin path below has isolated per-request failures
                # since it was written; the batched path had no such guard, so a
                # single request that ran the GPU out of memory took the worker
                # down and the supervisor then stopped the whole API server. A
                # model too large for Metal's working set made that reachable
                # from any prompt long enough to fill a prefill batch.
                self._abort_batch(reply, exc)
            self._reply(reply)
            return
        for req in list(self.active.values()):
            try:
                token, _logprobs = next(req.generator)
            except StopIteration:  # defensive: we never set a generator-side limit
                self._finish(req, reply, next_token=None, finish_reason="length")
                continue
            except Exception as exc:  # noqa: BLE001 -- isolate: one request, not the worker
                logger.warning(f"generation failed for request {req.uid}: {exc!r}")
                del self.active[req.uid]
                reply.append(ErrorReplyMsg(uid=req.uid, error=f"generation failed: {exc}"))
                continue
            next_token = int(token)
            req.output_ids.append(next_token)

            sp = req.sampling_params
            hit_length = (
                len(req.output_ids) >= sp.max_tokens
                or req.prompt_len + len(req.output_ids) >= self.max_seq_len
            )
            hit_eos = not sp.ignore_eos and next_token in self.eos_token_ids
            matched_stop = self._match_stop_str(req) if not hit_eos else None
            finished = hit_length or hit_eos or matched_stop is not None
            finish_reason = (
                ("stop" if (hit_eos or matched_stop is not None) else "length")
                if finished
                else None
            )
            reply.append(
                DetokenizeMsg(
                    uid=req.uid,
                    next_token=next_token,
                    finished=finished,
                    finish_reason=finish_reason,
                    matched_stop=matched_stop,
                    stop_strs=sp.stop_strs or None,
                    gpu_mem_bytes=gpu_mem,
                )
            )
            if finished:
                self._remember(req)
                del self.active[req.uid]
        self._reply(reply)

    def _step_batched(self, reply: List[BaseTokenizerMsg], gpu_mem: int) -> None:
        """One continuous-batching round: at most one decode token per active
        request plus a slice of prompt processing, all in batched forwards."""
        import os as _os
        import time as _time

        trace = _os.environ.get("MAXTOKEN_MLX_TRACE") == "1"
        t0 = _time.perf_counter() if trace else 0.0
        prompt_resps, gen_resps = self.batch_gen.next()
        if trace:
            t1 = _time.perf_counter()
            self._mx.synchronize()
            t2 = _time.perf_counter()
            outside = 1e3 * (t0 - self._trace_prev) if hasattr(self, "_trace_prev") else 0
            self._trace_prev = t2
            logger.info(
                f"trace: next()={1e3*(t1-t0):.1f}ms sync={1e3*(t2-t1):.1f}ms "
                f"outside={outside:.1f}ms B={len(gen_resps)} P={len(prompt_resps)}"
            )

        if self.prefix_store is not None:
            for pr in prompt_resps:
                # Mid-prompt segment boundaries are the prefix store's restore
                # points; the final split into generation is covered by the
                # end-of-request donation instead.
                if pr.end_of_segment and not pr.end_of_prompt:
                    uid = self._our_uid.get(pr.uid)
                    req = self.active.get(uid)
                    if req is None:
                        continue
                    done = pr.progress[0]
                    cached_prefix = len(req.prompt_ids) - pr.progress[1]
                    n = cached_prefix + done
                    extracted = self.batch_gen.extract_cache([pr.uid]).get(pr.uid)
                    if extracted is not None:
                        self.prefix_store.insert(req.prompt_ids[:n], extracted[0])

        for r in gen_resps:
            uid = self._our_uid.get(r.uid)
            req = self.active.get(uid)
            if req is None:
                continue
            next_token = int(r.token)
            req.output_ids.append(next_token)
            sp = req.sampling_params
            hit_eos = not sp.ignore_eos and next_token in self.eos_token_ids
            matched_stop = self._match_stop_str(req) if not hit_eos else None
            engine_done = r.finish_reason is not None  # "length" (cap or context)
            finished = engine_done or hit_eos or matched_stop is not None
            finish_reason = (
                ("stop" if (hit_eos or matched_stop is not None) else "length")
                if finished
                else None
            )
            reply.append(
                DetokenizeMsg(
                    uid=req.uid,
                    next_token=next_token,
                    finished=finished,
                    finish_reason=finish_reason,
                    matched_stop=matched_stop,
                    stop_strs=sp.stop_strs or None,
                    gpu_mem_bytes=gpu_mem,
                )
            )
            if finished:
                if engine_done:  # the engine already removed it and returned the cache
                    donated = r.prompt_cache
                else:
                    donated = self._batch_remove(req.uid)
                self._remember(req, donated)
                self._batch_forget(req.uid)
                del self.active[req.uid]

    def _batch_remove(self, uid: int):
        """Take a request out of the engine; returns its cache (or None)."""
        buid = self._batch_uid.get(uid)
        if buid is None:
            return None
        caches = self.batch_gen.remove([buid], return_prompt_caches=True)
        extracted = caches.get(buid)
        return extracted[0] if extracted else None

    def _batch_forget(self, uid: int) -> None:
        buid = self._batch_uid.pop(uid, None)
        if buid is not None:
            self._our_uid.pop(buid, None)

    def _remember(self, req: _MlxRequest, cache: Any = None) -> None:
        """Donate a finished/aborted request's cache to the prefix store. The
        cache covers the prompt plus all but the last emitted token (the last
        one was sampled but never fed back through the model)."""
        cache = cache if cache is not None else req.cache
        if self.prefix_store is None or cache is None:
            return
        tokens = req.prompt_ids + req.output_ids[:-1]
        # Speculative decode buffers several committed tokens per verify window;
        # a request that finishes mid-buffer (stop string, max_tokens) leaves the
        # cache ahead of what was emitted. Trim the overshoot where possible,
        # skip donation where not (recurrent caches cannot rewind).
        off = next(
            (c.offset for c in cache if c.is_trimmable() and hasattr(c, "offset")),
            None,
        )
        if off is not None and off != len(tokens):
            excess = off - len(tokens)
            if excess < 0 or not all(c.is_trimmable() for c in cache):
                return
            for c in cache:
                c.trim(excess)
        self.prefix_store.insert(tokens, cache)

    def _prefill_geometry(self, running: int) -> tuple[int, int]:
        """How wide and how deep a batched prefill forward may be.

        A prefill is the largest single allocation the engine makes: mlx-lm
        processes ``prefill_batch_size`` prompts at ``prefill_step_size`` tokens
        each in one forward. Metal reports a working set it is willing to keep
        resident — 24.96 GiB of the 32 GiB on this machine — and a checkpoint
        can approach or pass that on its own, at which point the default
        4x2048 has nothing left to allocate into and the command buffer aborts.

        So the geometry is read off what the weights actually left behind. The
        ladder is deliberately conservative: the true peak depends on the
        architecture's activation width, which is not something to guess at, and
        a batch that failed halves itself in ``_abort_batch`` regardless.
        """
        mx = self._mx
        try:
            info = mx.device_info()
            working_set = int(info.get("max_recommended_working_set_size", 0))
        except Exception:  # noqa: BLE001 -- non-Metal or a future API change
            working_set = 0
        weights = int(mx.get_active_memory())
        headroom = working_set - weights if working_set else 1 << 62
        gib = headroom / 2**30
        if headroom < 2 * 2**30:
            geometry = (1, 512)
        elif headroom < 6 * 2**30:
            geometry = (min(2, running), 1024)
        else:
            geometry = (min(4, running), 2048)
        logger.info(
            f"prefill geometry {geometry[0]}x{geometry[1]} tokens "
            f"(weights {weights / 2**30:.1f} GiB, working set headroom {gib:.1f} GiB)"
        )
        return geometry

    def _make_batch_generator(self):
        from mlx_lm.generate import BatchGenerator

        return BatchGenerator(
            self.model,
            completion_batch_size=self._batch_running,
            prefill_batch_size=self._prefill_batch,
            prefill_step_size=self._prefill_step,
        )

    def _abort_batch(self, reply: List[BaseTokenizerMsg], exc: BaseException) -> None:
        """Fail everything in flight and rebuild the batcher, then keep serving.

        A batched forward carries every active request, so there is no way to
        tell which one caused the failure — all of them are answered with an
        error. The generator itself is discarded rather than reused: a Metal
        command buffer that aborted leaves its prompt cache half-written, and
        the next call would fail on state from the request that already died.

        Freeing that state is also what makes the retry viable, which matters
        most for the case that gets here: a model whose weights alone approach
        the GPU's working set, where the peak of a prefill batch is what tips
        it over.
        """
        logger.warning(f"batched step failed, dropping {len(self.active)} request(s): {exc!r}")
        for uid in list(self.active):
            del self.active[uid]
            reply.append(ErrorReplyMsg(uid=uid, error=f"generation failed: {exc}"))
        self._batch_uid.clear()
        self._our_uid.clear()
        self.batch_gen = None
        self._mx.clear_cache()
        # Whatever the estimate was, this shape has now been shown not to fit.
        # Halving is what makes the next request succeed rather than repeat the
        # crash, and it is bounded: one prompt at 256 tokens a step is the
        # narrowest a batched prefill can be.
        if self._prefill_batch > 1 or self._prefill_step > 256:
            self._prefill_batch = max(1, self._prefill_batch // 2)
            self._prefill_step = max(256, self._prefill_step // 2)
            logger.warning(
                f"reducing prefill geometry to {self._prefill_batch}x{self._prefill_step} tokens"
            )
        try:
            self.batch_gen = self._make_batch_generator()
        except Exception as rebuild_exc:  # noqa: BLE001
            # Without a batcher the scheduler would busy-loop on every future
            # request; better to let the supervisor restart a worker that has
            # no way back.
            logger.error(f"could not rebuild the batcher: {rebuild_exc!r}")
            raise

    def _finish(
        self,
        req: _MlxRequest,
        reply: List[BaseTokenizerMsg],
        next_token: int | None,
        finish_reason: str,
    ) -> None:
        self._remember(req)
        del self.active[req.uid]
        if next_token is None:
            # No token to carry the terminal signal on: use an eos id so the
            # detokenizer drops it from the rendered text (see DetokenizeManager).
            next_token = next(iter(self.eos_token_ids), 0)
        reply.append(
            DetokenizeMsg(
                uid=req.uid,
                next_token=next_token,
                finished=True,
                finish_reason=finish_reason,
                stop_strs=req.sampling_params.stop_strs or None,
            )
        )

    # ------------------------------------------------------------------ msg handling

    def _handle_user_msg(self, msg: UserMsg) -> List[BaseTokenizerMsg]:
        input_ids = msg.input_ids.tolist()
        if len(input_ids) >= self.max_seq_len:
            return [
                ErrorReplyMsg(
                    uid=msg.uid,
                    error=(
                        f"prompt is {len(input_ids)} tokens but the model serves at most "
                        f"{self.max_seq_len}"
                    ),
                    code="context_length_exceeded",
                )
            ]
        try:
            if self.batch_gen is not None:
                cached = self._batch_admit(msg.uid, input_ids, msg.sampling_params)
                generator, cache = None, None
            else:
                generator, cache, cached = self._make_generator(
                    input_ids, msg.sampling_params
                )
        except Exception as exc:  # noqa: BLE001 -- surface as a request error, keep serving
            logger.warning(f"could not start request {msg.uid}: {exc!r}")
            return [ErrorReplyMsg(uid=msg.uid, error=f"could not start generation: {exc}")]
        self.active[msg.uid] = _MlxRequest(
            uid=msg.uid,
            sampling_params=msg.sampling_params,
            generator=generator,
            prompt_len=len(input_ids),
            prompt_ids=input_ids,
            cache=cache,
        )
        return [
            PromptAdmittedMsg(
                uid=msg.uid, prompt_tokens=len(input_ids), cached_tokens=cached
            )
        ]

    def _batch_admit(self, uid: int, input_ids: List[int], sp: SamplingParams) -> int:
        """Insert a request into the continuous-batching engine; returns the
        prefix-cache hit length. The prompt remainder is segmented at snapshot
        boundaries so the engine pauses there and mid-prompt states can be
        donated to the prefix store."""
        import os as _os

        from .prefix_cache import BOUNDARY_TOKENS

        cache, cached = self._lookup_prefix(input_ids)
        # Experimental, opt-in (MAXTOKEN_MLX_PREFETCH=1): madvise the mapped
        # store in ahead of a big prefill. Measured NET-NEGATIVE on this class
        # of hardware (cold 144 -> 131 tok/s, warm 399 -> 264): without a
        # per-layer progress hook the sweep competes with the forward's own
        # demand faults for the SSD queue instead of running ahead of them.
        # Kept for machines where the tradeoff may differ; prefer
        # MAXTOKEN_MLX_MLOCK=1, which pins the store and preloads it at start.
        store = getattr(self.model, "_maxtoken_mapped_store", None)
        if (
            store is not None
            and len(input_ids) - cached >= 256
            and _os.environ.get("MAXTOKEN_MLX_PREFETCH") == "1"
        ):
            store.start_prefetch()
        segments = []
        pos = cached
        while pos < len(input_ids):
            end = min(len(input_ids), (pos // BOUNDARY_TOKENS + 1) * BOUNDARY_TOKENS)
            segments.append(input_ids[pos:end])
            pos = end
        # Output budget: the request's own cap plus the model context ceiling;
        # the engine reports "length" and hands the cache back at the boundary.
        cap = max(1, min(sp.max_tokens, self.max_seq_len - len(input_ids)))
        buid = self.batch_gen.insert_segments(
            [segments],
            max_tokens=[cap],
            caches=[cache],
            samplers=[self._build_sampler(sp)],
        )[0]
        self._batch_uid[uid] = buid
        self._our_uid[buid] = uid
        return cached

    def _handle(self, msg: BaseBackendMsg) -> tuple[List[BaseTokenizerMsg], bool]:
        """Returns (replies, exit_requested)."""
        if isinstance(msg, BatchBackendMsg):
            replies: List[BaseTokenizerMsg] = []
            for m in msg.data:
                r, do_exit = self._handle(m)
                replies.extend(r)
                if do_exit:
                    return replies, True
            return replies, False
        if isinstance(msg, UserMsg):
            return self._handle_user_msg(msg), False
        if isinstance(msg, AbortBackendMsg):
            # Terminal for the uid; the frontend's abort ack is the client-facing reply,
            # so no message goes back from here (same as the CUDA scheduler).
            req = self.active.pop(msg.uid, None)
            if req is not None:
                donated = self._batch_remove(msg.uid) if self.batch_gen else None
                self._batch_forget(msg.uid)
                self._remember(req, donated)
            return [], False
        if isinstance(msg, CacheRebuildBackendMsg):
            return [self._handle_cache_rebuild(msg)], False
        if isinstance(msg, ExitMsg):
            return [], True
        logger.warning(f"MLX scheduler ignoring unexpected message {type(msg).__name__}")
        return [], False

    def _handle_cache_rebuild(self, msg: CacheRebuildBackendMsg) -> BaseTokenizerMsg:
        """Elastic memory management, the runtime resize: change the expert
        slot-cache size and/or the context-window ceiling without restarting or
        reloading. KV is per-request on MLX, so ``max_seq_len`` (not a page
        pool) is this backend's capacity knob: admission and generation caps
        follow it immediately."""
        wants_moe = bool(msg.moe_cache_size)
        wants_ctx = bool(getattr(msg, "max_seq_len", None))
        if not wants_moe and not wants_ctx:
            return CacheRebuildResultMsg(
                request_id=msg.request_id,
                status="failed",
                error="the MLX backend can rebuild moe_cache_size and/or "
                "max_seq_len (KV is per-request, not pooled)",
            )
        if wants_moe and self.offload_state is None:
            return CacheRebuildResultMsg(
                request_id=msg.request_id,
                status="failed",
                error="no expert cache to rebuild (serving fully resident); "
                "start with --moe-backend offload",
            )
        if self.active:
            return CacheRebuildResultMsg(request_id=msg.request_id, status="busy")
        total = 0
        if wants_moe:
            total = self.offload_state.resize_total(int(msg.moe_cache_size))
            logger.info(
                f"expert cache resized to {total} slots "
                f"({self.offload_state.cache_bytes() / 2**30:.2f} GiB)"
            )
            if self.draft is not None:
                # A shrunk cache may no longer hold a whole verify window.
                self.draft.k = self._clamp_draft_k(self.draft.k)
                self.offload_state.spec_window = self.draft.k + 1
        if wants_ctx:
            self.max_seq_len = max(1024, int(msg.max_seq_len))
            logger.info(f"context window ceiling set to {self.max_seq_len} tokens")
        return CacheRebuildResultMsg(
            request_id=msg.request_id,
            status="ok",
            moe_cache_size=total,
            max_seq_len=self.max_seq_len if wants_ctx else 0,
        )

    def _reply(self, replies: List[BaseTokenizerMsg]) -> None:
        import os as _os

        if _os.environ.get("MAXTOKEN_MLX_NO_REPLY") == "1":  # diagnosis only
            return
        if len(replies) == 1:
            self._send.put(replies[0])
        elif len(replies) > 1:
            self._send.put(BatchTokenizerMsg(data=replies))

    # ------------------------------------------------------------------ main loop

    def run_forever(self) -> None:
        try:
            while True:
                pending: List[BaseBackendMsg] = []
                if not self.active:
                    self._maybe_rebalance()  # idle: safe to rebuild slot buffers
                    pending.append(self._recv.get())  # idle: block for work
                while not self._recv.empty():
                    pending.append(self._recv.get())
                replies: List[BaseTokenizerMsg] = []
                for msg in pending:
                    r, do_exit = self._handle(msg)
                    replies.extend(r)
                    if do_exit:
                        self._reply(replies)
                        return
                self._reply(replies)
                if self.active:
                    self._step()
        except KeyboardInterrupt:
            pass

    def shutdown(self) -> None:
        self.active.clear()
        self._recv.stop()
        self._send.stop()


def _wrap_fast_quantized(load):
    """Wrap mlx-lm's ``load`` so a big quantized checkpoint can be opened at all.

    mlx-lm builds a quantized model by constructing RANDOM float weights,
    quantizing them, and only then overwriting everything from disk. That
    middle step is pure waste and its transient peaks near the full model
    size — a 100B-class quant is OS-killed inside ``load()`` on a 32 GiB Mac
    (no traceback, just a dead process).

    The ``mlx-optiq`` package ships a context manager that installs the
    checkpoint's own lazy arrays into the quantized modules instead, so
    nothing is allocated. Use it when it is importable; without it, load
    unchanged (models that fit are unaffected either way).
    """
    try:
        from optiq.runtime.fast_load import fast_quantized_load
    except Exception:  # noqa: BLE001 -- optional; plain load is the fallback
        return load

    def _load(*args, **kwargs):
        with fast_quantized_load():
            return load(*args, **kwargs)

    logger.info("quantized load: installing checkpoint arrays directly (mlx-optiq)")
    return _load


def _preimport_architectures() -> None:
    """Import modules that register out-of-tree model architectures with mlx-lm
    before the model is loaded (MAXTOKEN_MLX_PREIMPORT, comma-separated).

    mlx-lm dispatches on ``model_type`` through its own package, so an
    architecture it does not ship cannot be served — even when a third-party
    package implements it. Those packages register themselves on import (e.g.
    ``import optiq`` adds deepseek_v4); naming them here makes their models
    loadable without vendoring anyone's model code.
    """
    import importlib
    import os

    for name in os.environ.get("MAXTOKEN_MLX_PREIMPORT", "").split(","):
        name = name.strip()
        if not name:
            continue
        try:
            importlib.import_module(name)
            logger.info(f"pre-imported {name} (registers extra model architectures)")
        except Exception as exc:  # noqa: BLE001 -- a bad name must not kill the worker
            logger.warning(f"MAXTOKEN_MLX_PREIMPORT: could not import {name}: {exc}")


def _raise_qos() -> None:
    """Promote this thread's macOS QoS class before MLX spawns its workers.

    A worker spawned from the server's launcher can inherit a throttled QoS
    band; the threads MLX creates then run at reduced priority (observed as
    priority-20 threads doing all the compute), which serializes the decode
    pipeline and roughly doubles batched step time. Threads inherit the QoS of
    the thread that creates them, so raising the main thread FIRST fixes every
    MLX thread spawned afterwards."""
    import ctypes
    import sys as _sys

    if _sys.platform != "darwin":
        return
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        QOS_CLASS_USER_INTERACTIVE = 0x21
        libsystem.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    except Exception:  # noqa: BLE001 -- a nicety; never block startup on it
        pass


def mlx_scheduler_worker(config: SchedulerConfig, ack_queue: Any) -> None:
    """Process target: build the scheduler, ack readiness, serve until exit.

    Ack protocol matches ``launch._run_scheduler``: optional ("progress", …) ticks
    while loading, a single readiness string, and ("error", reason) before dying on
    a startup failure so the supervisor can report the real cause.
    """
    _raise_qos()
    try:
        ack_queue.put(("progress", "Loading weights (MLX)", 0, 0))
        scheduler = MlxScheduler(config)
    except Exception as exc:  # noqa: BLE001 -- report, then let it propagate
        try:
            ack_queue.put(("error", f"{type(exc).__name__}: {exc}"))
            ack_queue.close()
            ack_queue.join_thread()
        except Exception:  # noqa: BLE001 -- reporting must never mask the failure
            pass
        raise
    if scheduler.offload_state is not None:
        # Cache-geometry meta for the frontend's cache panel (best-effort, same
        # contract as the CUDA engine's readiness meta: "pools" + unit bytes).
        try:
            st = scheduler.offload_state
            _h, _m, slots = st.totals()
            ack_queue.put(
                (
                    "meta",
                    {
                        "moe_bytes_per_expert": st.glus[0].store.expert_nbytes,
                        "pools": {
                            "moe_cache_size": slots,
                            "num_pages": 0,
                            "page_size": 1,
                            "num_mamba_slots": 0,
                        },
                    },
                )
            )
        except Exception:  # noqa: BLE001 -- metadata is a nicety
            pass
    ack_queue.put("Scheduler is ready")
    import os as _os

    profile_path = _os.environ.get("MAXTOKEN_MLX_PROFILE")
    try:
        if profile_path:
            import cProfile
            import signal

            def _graceful(_sig, _frm):  # let finally run so the dump is written
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, _graceful)
            prof = cProfile.Profile()
            try:
                prof.runcall(scheduler.run_forever)
            finally:
                prof.dump_stats(profile_path)
        else:
            scheduler.run_forever()
    finally:
        scheduler.shutdown()
