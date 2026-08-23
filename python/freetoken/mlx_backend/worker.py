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
        self.model, self.tokenizer = load(config.model_path)
        hf_tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.eos_token_ids = frozenset(load_eos_token_ids(config.model_path, hf_tokenizer))
        self.config = config
        self.max_seq_len = int(config.max_seq_len)

        self._recv = ZmqPullQueue(
            config.zmq_backend_addr, create=True, decoder=BaseBackendMsg.decoder
        )
        self._send = ZmqPushQueue(
            config.zmq_detokenizer_addr,
            create=config.backend_create_detokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        )
        self.active: Dict[int, _MlxRequest] = {}

    # ------------------------------------------------------------------ generation

    def _make_generator(self, input_ids: List[int], sp: SamplingParams) -> Iterator[Any]:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        mx = self._mx
        sampler = None
        if not sp.is_greedy:
            sampler = make_sampler(
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
        # max_tokens=-1 -> unbounded; EOS/length/stop are all enforced in _step so
        # ignore_eos and the exact CUDA-scheduler semantics stay in one place.
        kwargs = _filter_kwargs(
            generate_step, {"max_tokens": -1, "sampler": sampler}
        )
        return generate_step(mx.array(input_ids), self.model, **kwargs)

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
                del self.active[req.uid]
        self._reply(reply)

    def _finish(
        self,
        req: _MlxRequest,
        reply: List[BaseTokenizerMsg],
        next_token: int | None,
        finish_reason: str,
    ) -> None:
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
            generator = self._make_generator(input_ids, msg.sampling_params)
        except Exception as exc:  # noqa: BLE001 -- surface as a request error, keep serving
            logger.warning(f"could not start request {msg.uid}: {exc!r}")
            return [ErrorReplyMsg(uid=msg.uid, error=f"could not start generation: {exc}")]
        self.active[msg.uid] = _MlxRequest(
            uid=msg.uid,
            sampling_params=msg.sampling_params,
            generator=generator,
            prompt_len=len(input_ids),
        )
        return [PromptAdmittedMsg(uid=msg.uid, prompt_tokens=len(input_ids), cached_tokens=0)]

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
            self.active.pop(msg.uid, None)
            return [], False
        if isinstance(msg, CacheRebuildBackendMsg):
            return [
                CacheRebuildResultMsg(
                    request_id=msg.request_id,
                    status="failed",
                    error="cache rebuild is not supported by the MLX backend",
                )
            ], False
        if isinstance(msg, ExitMsg):
            return [], True
        logger.warning(f"MLX scheduler ignoring unexpected message {type(msg).__name__}")
        return [], False

    def _reply(self, replies: List[BaseTokenizerMsg]) -> None:
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


def mlx_scheduler_worker(config: SchedulerConfig, ack_queue: Any) -> None:
    """Process target: build the scheduler, ack readiness, serve until exit.

    Ack protocol matches ``launch._run_scheduler``: optional ("progress", …) ticks
    while loading, a single readiness string, and ("error", reason) before dying on
    a startup failure so the supervisor can report the real cause.
    """
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
    try:
        scheduler.run_forever()
    finally:
        scheduler.shutdown()
