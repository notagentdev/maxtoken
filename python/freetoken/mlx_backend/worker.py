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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

from freetoken.message import (
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
from freetoken.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_eos_token_ids

if TYPE_CHECKING:
    from freetoken.core import SamplingParams
    from freetoken.scheduler import SchedulerConfig

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
        import mlx.core as mx  # noqa: F401 -- fail here, before any socket binds
        from mlx_lm import load

        self._mx = mx
        # mlx-lm resolves both local paths and hub ids (through the HF cache),
        # matching the tokenizer workers' resolution.
        logger.info(f"Loading MLX model from {config.model_path}")
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
        # Continuous batching (resident and mapped-expert serving): concurrent
        # requests decode in ONE batched forward per step instead of one forward
        # per request per token. The slot-cache offload path keeps its own
        # speculate/verify loop and stays round-robin.
        self.batch_gen = None
        self._batch_uid: dict[int, int] = {}  # our uid -> engine uid
        self._our_uid: dict[int, int] = {}  # engine uid -> our uid
        if self.offload_state is None:
            from mlx_lm.generate import BatchGenerator

            self.batch_gen = BatchGenerator(
                self.model,
                completion_batch_size=max(1, config.max_running_req),
                prefill_batch_size=min(4, max(1, config.max_running_req)),
            )
            logger.info(
                f"continuous batching: up to {max(1, config.max_running_req)} "
                "concurrent decodes per forward"
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

            budget = int(_os.environ.get("FREETOKEN_MLX_PREFIX_CACHE_MB", "0")) * 2**20
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
        """Wire the expert slot cache in (FreeToken's core: serve a model whose
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
        return make_sampler(
            **_filter_kwargs(
                make_sampler,
                {
                    "temp": max(sp.temperature, 0.0),
                    # mlx-lm encodes "disabled" as 0 (freetoken: 1.0 / -1).
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

    def _prefill_into(self, cache, input_ids: List[int], start: int) -> None:
        """Process input_ids[start:-1] into ``cache`` in chunks, snapshotting at
        BOUNDARY_TOKENS multiples so hybrid models (whose recurrent state cannot
        be trimmed) have exact restore points for future prefix hits."""
        from .prefix_cache import BOUNDARY_TOKENS

        mx = self._mx
        pos = start
        end = len(input_ids) - 1
        next_boundary = (pos // BOUNDARY_TOKENS + 1) * BOUNDARY_TOKENS
        while pos < end:
            n = min(2048, end - pos, next_boundary - pos)
            logits = self.model(mx.array(input_ids[pos:pos + n])[None], cache=cache)
            mx.eval(logits)
            pos += n
            if pos == next_boundary and pos < end and self.prefix_store is not None:
                self.prefix_store.insert(input_ids[:pos], cache)
                next_boundary += BOUNDARY_TOKENS

    def _make_generator(self, input_ids: List[int], sp: SamplingParams) -> tuple:
        """(token generator, live cache list | None, cached prefix tokens)."""
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        mx = self._mx
        cache, cached = self._lookup_prefix(input_ids)
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
        """Cheap per-step rollback point for one mlx-lm cache object. Trimmable
        caches (KV) roll back by rewinding their offset; recurrent caches (GDN
        conv/state) roll back by restoring the previous arrays — mx arrays are
        immutable, so holding the refs is enough."""
        if c.is_trimmable():
            return None
        return list(c.state)

    @staticmethod
    def _cache_rollback(c, snap) -> None:
        if snap is None:
            c.trim(1)
        else:
            c.state = snap

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
        from .prefix_cache import BOUNDARY_TOKENS

        mx = self._mx
        state = self.offload_state
        sampler = self._build_sampler(sp)

        # Prefill (all tokens but the last): per-layer sync/streamed serving,
        # with prefix-store snapshots at restore-safe boundaries.
        state.speculating = False
        pos, end = start, len(input_ids) - 1
        next_boundary = (pos // BOUNDARY_TOKENS + 1) * BOUNDARY_TOKENS
        while pos < end:
            n = min(2048, end - pos, next_boundary - pos)
            state.begin_token()
            logits = self.model(mx.array(input_ids[pos:pos + n])[None], cache=cache)
            mx.eval(logits)
            pos += n
            if pos == next_boundary and pos < end and self.prefix_store is not None:
                self.prefix_store.insert(input_ids[:pos], cache)
                next_boundary += BOUNDARY_TOKENS
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
                logger.info(
                    f"expert cache: {slots} slots, lifetime miss rate {rate:.1%} "
                    f"({m}/{h + m}), active mem {gpu_mem / 2**30:.2f} GiB"
                )
        if self.batch_gen is not None:
            self._step_batched(reply, gpu_mem)
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

        trace = _os.environ.get("FREETOKEN_MLX_TRACE") == "1"
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
        self.prefix_store.insert(tokens, cache)

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
        # Experimental, opt-in (FREETOKEN_MLX_PREFETCH=1): madvise the mapped
        # store in ahead of a big prefill. Measured NET-NEGATIVE on this class
        # of hardware (cold 144 -> 131 tok/s, warm 399 -> 264): without a
        # per-layer progress hook the sweep competes with the forward's own
        # demand faults for the SSD queue instead of running ahead of them.
        # Kept for machines where the tradeoff may differ; prefer
        # FREETOKEN_MLX_MLOCK=1, which pins the store and preloads it at start.
        store = getattr(self.model, "_freetoken_mapped_store", None)
        if (
            store is not None
            and len(input_ids) - cached >= 256
            and _os.environ.get("FREETOKEN_MLX_PREFETCH") == "1"
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
        """Elastic memory management, FreeToken's runtime cache resize: change the
        expert slot-cache size without restarting or reloading the model."""
        if self.offload_state is None:
            return CacheRebuildResultMsg(
                request_id=msg.request_id,
                status="failed",
                error="no expert cache to rebuild (serving fully resident); "
                "start with --moe-backend offload",
            )
        if not msg.moe_cache_size:
            return CacheRebuildResultMsg(
                request_id=msg.request_id,
                status="failed",
                error="the MLX backend can only rebuild moe_cache_size "
                "(KV is per-request, not pooled)",
            )
        if self.active:
            return CacheRebuildResultMsg(request_id=msg.request_id, status="busy")
        total = self.offload_state.resize_total(int(msg.moe_cache_size))
        logger.info(
            f"expert cache resized to {total} slots "
            f"({self.offload_state.cache_bytes() / 2**30:.2f} GiB)"
        )
        return CacheRebuildResultMsg(
            request_id=msg.request_id, status="ok", moe_cache_size=total
        )

    def _reply(self, replies: List[BaseTokenizerMsg]) -> None:
        import os as _os

        if _os.environ.get("FREETOKEN_MLX_NO_REPLY") == "1":  # diagnosis only
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
    ack_queue.put("Scheduler is ready")
    import os as _os

    profile_path = _os.environ.get("FREETOKEN_MLX_PROFILE")
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
