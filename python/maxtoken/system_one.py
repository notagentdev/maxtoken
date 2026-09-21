"""System One decisions: typed, calibrated answers from a local MLX model.

The model never writes prose. A question carries its own answer set, the
prompt is scored against exactly that set, and what comes back is a
probability distribution over it. Structured-output errors are impossible by
construction because nothing outside the set can be returned.

Answers are read from uppercase letter slots: the options are listed as
``A``, ``B``, ``C`` ... in the prompt and only those single letter tokens are
scored. That keeps every candidate exactly one token, so a question costs one
forward pass however long its options are, and it removes the prefix problem
that bare answer strings have (``"1"`` is a token prefix of ``"10"`` and would
otherwise absorb its probability). Candidate sets too large for the letters
fall back to scoring the answer strings directly, closed with the end-of-turn
token so the set stays mutually exclusive.

The expensive part of a decision is the prefill of the shared state, so it is
paid once: all questions for one state are tokenized, their longest common
token prefix is prefilled into a single cache, and every question runs from a
fork of it. Answering ten questions about one ticket costs one prefill plus
ten short suffixes, not ten prefills.

Question specs mirror the shape that has become conventional for this kind of
API::

    {"category": {"type": "choice", "options": ["bug", "feature", "noise"]},
     "urgency":  {"type": "score", "min": 0, "max": 100},
     "actionable": {"type": "bool"}}

Answers carry the chosen value, its probability, and the full distribution.
Unlike a hosted decision model the probabilities here are ours to inspect and
to correct: `fit_temperature` refits a single scalar on labelled cases and
`ece` reports what that refit bought. Both measure the workload they were run
on; a softmax over the allowed answers is conditional on exactly those
alternatives and is not operational confidence anywhere else.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache

#: Answer slots. Sixteen is as many options as one letter round-trips for.
LETTERS = "ABCDEFGHIJKLMNOP"

#: Questions with more candidates than this are thinned to evenly spaced
#: anchors; a 0-100 score does not need 101 forward passes to be calibrated.
MAX_CANDIDATES = 64

DEFAULT_SYSTEM = (
    "Apply the question to the supplied state and choose exactly one listed "
    "option. Respond with only its uppercase letter, with no explanation."
)


@dataclass
class Answer:
    """One typed decision plus the distribution it was drawn from."""

    value: Any
    probability: float
    distribution: Dict[str, float]
    entropy: float
    #: Expected value over the distribution; only meaningful for scores.
    expectation: Optional[float] = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Answer({self.value!r}, p={self.probability:.3f})"


@dataclass
class Decision:
    """Every answer for one state, with the timings that produced them."""

    answers: Dict[str, Answer]
    prefill_tokens: int
    prefill_ms: float
    decide_ms: float

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    @property
    def values(self) -> Dict[str, Any]:
        return {k: a.value for k, a in self.answers.items()}


def _fork(cache: Sequence[Any]) -> List[Any]:
    """Copy a prompt cache so a question can extend it without disturbing it.

    A `KVCache` restored through `from_state` reports `offset == keys.shape[2]`,
    so its next write reallocates instead of writing into the buffer we came
    from — the prefix is shared until something would overwrite it. Recurrent
    caches keep small tensors that *are* updated in place, so those are copied
    outright.
    """
    forked = []
    for c in cache:
        state = c.state
        if not isinstance(c, KVCache):
            state = _copy_tree(state)
        forked.append(type(c).from_state(state, c.meta_state))
    return forked


def _copy_tree(value: Any) -> Any:
    if isinstance(value, mx.array):
        return mx.zeros_like(value) + value
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_tree(v) for v in value)
    return value


def _prefix_free(token_lists: Sequence[Sequence[int]]) -> bool:
    """True when no candidate's tokens are a strict prefix of another's."""
    seen = {tuple(t) for t in token_lists}
    return not any(
        tuple(t[:i]) in seen for t in token_lists for i in range(1, len(t))
    )


def _expand(cache: Sequence[Any], batch: int) -> List[Any]:
    """Replicate a batch-1 prompt cache across `batch` rows."""
    out = []
    for c in cache:
        state = c.state
        if isinstance(state, (list, tuple)):
            grown = type(state)(
                mx.repeat(a, batch, axis=0) if isinstance(a, mx.array) and a.shape[0] == 1
                else a
                for a in state
            )
        else:
            grown = state
        out.append(type(c).from_state(grown, c.meta_state))
    return out


def _common_prefix(sequences: Sequence[Sequence[int]]) -> int:
    if not sequences:
        return 0
    shortest = min(len(s) for s in sequences)
    first = sequences[0]
    for i in range(shortest):
        if any(s[i] != first[i] for s in sequences[1:]):
            return i
    return shortest


def _thin(values: List[Any], limit: int) -> List[Any]:
    if len(values) <= limit:
        return values
    step = (len(values) - 1) / (limit - 1)
    picked = [values[round(i * step)] for i in range(limit)]
    seen, out = set(), []
    for v in picked:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@dataclass
class Question:
    """A question together with the exact set of answers it permits."""

    name: str
    prompt: str
    #: What the options mean, as shown in the prompt.
    descriptions: List[str]
    #: The value each option stands for.
    values: List[Any]
    numeric: bool = False
    #: Letter slots, when the option count allows them.
    slots: bool = True

    @classmethod
    def from_spec(cls, name: str, spec: Dict[str, Any],
                  use_slots: bool = True) -> "Question":
        kind = spec.get("type", "choice")
        prompt = spec.get("prompt", name)
        limit = len(LETTERS) if use_slots else MAX_CANDIDATES
        if kind == "choice":
            values = list(spec["options"])
        elif kind == "bool":
            labels = list(spec.get("labels", ("yes", "no")))
            q = cls(name, prompt, [str(l) for l in labels], [True, False])
            q.slots = use_slots
            return q
        elif kind == "score":
            lo, hi = int(spec["min"]), int(spec["max"])
            step = int(spec.get("step", 1))
            values = _thin(list(range(lo, hi + 1, step)), limit)
            q = cls(name, prompt, [str(v) for v in values], values, numeric=True)
            q.slots = use_slots and len(values) <= len(LETTERS)
            return q
        else:
            raise ValueError(f"unknown question type {kind!r}")
        if len(values) > limit:
            raise ValueError(f"{name}: {len(values)} options exceed the limit {limit}")
        q = cls(name, prompt, [str(v) for v in values], values)
        q.slots = use_slots and len(values) <= len(LETTERS)
        return q

    def render(self) -> str:
        # The format instruction has to come *after* the options: put it
        # before them and the model reaches for markdown scaffolding instead
        # of an answer, which leaves the restricted softmax normalising over
        # noise. Never pre-seed an "Answer:" prefix either — it reads as a
        # numbered list and the next token becomes a digit.
        if self.slots:
            options = "\n".join(
                f"{LETTERS[i]}. {d}" for i, d in enumerate(self.descriptions)
            )
            return (f"Question: {self.prompt}\nOptions:\n{options}\n"
                    "Reply with only the uppercase letter of your choice.")
        options = " | ".join(self.descriptions)
        return (f"Question: {self.prompt}\nAllowed answers: {options}\n"
                "Reply with exactly one of the allowed answers and nothing else.")


class SystemOne:
    """A local decision head over any mlx-lm model."""

    def __init__(self, model, tokenizer, *, system: str = DEFAULT_SYSTEM,
                 temperature: float = 1.0, use_slots: bool = True,
                 batch: bool = True, batch_waste: float = 1.3):
        self.model = model
        self.tokenizer = tokenizer
        self.system = system
        #: Calibration temperature applied to the restricted logits. 1.0 is
        #: the raw model; `fit_temperature` replaces it with a fitted value.
        self.temperature = temperature
        self.use_slots = use_slots
        #: Score every question for one state in a single forward pass.
        self.batch = batch
        #: Padded/real token ratio above which batching stops paying.
        self.batch_waste = batch_waste
        self._candidate_tokens: Dict[str, List[int]] = {}
        self._slot_cache: Dict[int, List[int]] = {}
        self._boundary_checked: set = set()

    @classmethod
    def load(cls, path: str, strict: bool = True, **kwargs) -> "SystemOne":
        """Load an mlx-lm model.

        `strict=False` tolerates weights the model class does not declare.
        That is only ever right when the implementation deliberately omits
        them — Gemma 4 shares KV across its last `num_kv_shared_layers`
        layers, so their `k_proj`/`v_proj` are dead weight that the converter
        copied across. Verify the model still answers before trusting a
        non-strict load.
        """
        from mlx_lm import load
        from mlx_lm.utils import hf_repo_to_path, load_model, load_tokenizer

        if strict:
            model, tokenizer = load(path)
            return cls(model, tokenizer, **kwargs)
        import pathlib as _pathlib
        local = _pathlib.Path(path)
        model_path = local if local.exists() else hf_repo_to_path(path)
        model, config = load_model(model_path, strict=False)
        tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id"))
        return cls(model, tokenizer, **kwargs)

    # -- answer slots ------------------------------------------------------

    def slot_ids(self, count: int) -> List[int]:
        """The token ids of the first `count` letters, verified as slots.

        A slot has to be one token that decodes back to itself, and the slots
        have to be distinct — otherwise the restricted softmax is scoring
        something other than the options.
        """
        cached = self._slot_cache.get(count)
        if cached is not None:
            return cached
        if count > len(LETTERS):
            raise ValueError(f"{count} options exceed the {len(LETTERS)} letter slots")
        ids = []
        for letter in LETTERS[:count]:
            encoded = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) != 1 or self.tokenizer.decode(encoded) != letter:
                raise ValueError(f"answer slot {letter!r} is not one exact round-trip token")
            ids.append(encoded[0])
        if len(set(ids)) != len(ids):
            raise ValueError("answer-slot tokens collide")
        self._slot_cache[count] = ids
        return ids

    def _check_boundary(self, text: str, ids: Sequence[int], slots: Sequence[int]) -> None:
        """Appending a slot must not re-tokenize the end of the prompt.

        Checked once per prompt ending and option count: if the last prompt
        token merges with the letter, the logits we read are not the slot's.
        """
        key = (text[-32:], len(slots))
        if key in self._boundary_checked:
            return
        for letter, slot in zip(LETTERS, slots):
            if self.tokenizer.encode(text + letter, add_special_tokens=False) != list(ids) + [slot]:
                raise ValueError(f"answer boundary changes tokenization for slot {letter}")
        self._boundary_checked.add(key)

    # -- prompt plumbing ---------------------------------------------------

    def _encode(self, state: str, question: Question):
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"{state}\n\n{question.render()}"},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Templates without a thinking switch.
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if question.slots:
            slots = self.slot_ids(len(question.descriptions))
            self._check_boundary(text, ids, slots)
        return ids

    def _tokens_for(self, candidate: str) -> List[int]:
        cached = self._candidate_tokens.get(candidate)
        if cached is None:
            cached = self.tokenizer.encode(candidate, add_special_tokens=False)
            self._candidate_tokens[candidate] = cached
        return cached

    def _stop_token(self) -> int:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        if eos is None:
            raise ValueError("tokenizer has no eos token to close candidates with")
        return int(eos)

    def _forward(self, tokens: Sequence[int], cache) -> mx.array:
        logits = self.model(mx.array(list(tokens))[None], cache=cache)
        logits = getattr(logits, "logits", logits)
        return logits[0].astype(mx.float32)

    # -- scoring -----------------------------------------------------------

    def _score(self, question: Question, cache, last_logits: mx.array) -> List[float]:
        """Total log-probability of each candidate continuation."""
        if question.slots:
            token_lists = [[s] for s in self.slot_ids(len(question.descriptions))]
        else:
            token_lists = [self._tokens_for(c) for c in question.descriptions]
            if not _prefix_free(token_lists):
                # "1" is a token prefix of "10", so scoring it as a bare token
                # hands it every continuation's mass and it wins whichever way
                # the model leans. Closing each candidate with the end-of-turn
                # token makes the set mutually exclusive again.
                stop = self._stop_token()
                token_lists = [list(t) + [stop] for t in token_lists]
        logprobs = last_logits - mx.logsumexp(last_logits, keepdims=True)

        if all(len(t) == 1 for t in token_lists):
            # The fast path: one forward pass answered the whole question.
            ids = mx.array([t[0] for t in token_lists])
            return logprobs[ids].tolist()

        scores = []
        for tokens in token_lists:
            total = float(logprobs[tokens[0]].item())
            if len(tokens) > 1:
                fork = _fork(cache)
                out = self._forward(tokens[:-1], fork)
                step = out - mx.logsumexp(out, axis=-1, keepdims=True)
                rows = mx.array(tokens[1:])
                total += float(
                    mx.take_along_axis(step, rows[:, None], axis=-1).sum().item()
                )
            scores.append(total)
        return scores

    def _answer(self, question: Question, scores: Sequence[float]) -> Answer:
        t = max(self.temperature, 1e-3)
        scaled = mx.array(list(scores)) / t
        probs = mx.softmax(scaled).tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        expectation = None
        if question.numeric:
            expectation = sum(p * float(v) for p, v in zip(probs, question.values))
        return Answer(
            value=question.values[best],
            probability=probs[best],
            distribution=dict(zip(question.descriptions, probs)),
            entropy=entropy,
            expectation=expectation,
        )

    # -- public API --------------------------------------------------------

    def decide(self, state: str, questions: Dict[str, Dict[str, Any]]) -> Decision:
        """Answer every question about `state`, sharing one prefill."""
        parsed = [Question.from_spec(n, s, self.use_slots)
                  for n, s in questions.items()]
        encoded = [self._encode(state, q) for q in parsed]
        shared = _common_prefix(encoded)

        t0 = time.perf_counter()
        cache = make_prompt_cache(self.model)
        if shared:
            # The last shared token is left to the per-question pass so that
            # every fork ends on a real forward and owns its final logits.
            self._forward(encoded[0][: shared - 1], cache)
            mx.eval([c.state for c in cache])
        t1 = time.perf_counter()

        start = max(shared - 1, 0)
        suffixes = [t[start:] for t in encoded]
        answers: Dict[str, Answer] = {}

        # Right padding makes every row as long as the longest, so a batch of
        # uneven questions buys parallelism with wasted tokens. Batch only
        # when the waste stays under `self.batch_waste`; a three-question set
        # where one lists eleven options is slower batched than sequential.
        real = sum(len(x) for x in suffixes)
        padded = len(suffixes) * max(len(x) for x in suffixes) if suffixes else 0
        worth_it = real > 0 and padded <= self.batch_waste * real

        if (self.batch and worth_it and len(parsed) > 1
                and all(q.slots for q in parsed)):
            # One forward pass for every question. Right padding is safe:
            # attention is causal and the recurrent layers are too, so tokens
            # after a row's last real one cannot reach the logits we read.
            # Each row is therefore scored exactly as if it had run alone.
            width = max(len(s) for s in suffixes)
            pad = self._stop_token()
            padded = mx.array([s + [pad] * (width - len(s)) for s in suffixes])
            fork = _expand(cache, len(suffixes))
            logits = self.model(padded, cache=fork)
            logits = getattr(logits, "logits", logits).astype(mx.float32)
            for row, (question, suffix) in enumerate(zip(parsed, suffixes)):
                last = logits[row, len(suffix) - 1]
                answers[question.name] = self._answer(
                    question, self._score(question, None, last)
                )
        else:
            for question, tokens in zip(parsed, suffixes):
                fork = _fork(cache)
                out = self._forward(tokens, fork)
                answers[question.name] = self._answer(
                    question, self._score(question, fork, out[-1])
                )
        t2 = time.perf_counter()

        return Decision(
            answers=answers,
            prefill_tokens=max(shared - 1, 0),
            prefill_ms=(t1 - t0) * 1000,
            decide_ms=(t2 - t1) * 1000,
        )


# -- calibration -----------------------------------------------------------


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error: mean gap between confidence and accuracy."""
    total = len(confidences)
    if total == 0:
        return 0.0
    error = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi or (b == 0 and c <= hi))]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1.0 for i in idx if correct[i]) / len(idx)
        error += (len(idx) / total) * abs(conf - acc)
    return error


def fit_temperature(scores: Sequence[Sequence[float]], labels: Sequence[int],
                    grid: Optional[Sequence[float]] = None) -> float:
    """Pick the temperature that minimises negative log-likelihood.

    `scores` holds the raw candidate log-probabilities per case and `labels`
    the index of the correct candidate. One scalar, fitted on held-out cases,
    is what turns a confident model into an honest one.
    """
    grid = grid or [0.25 + 0.05 * i for i in range(76)]
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = 0.0
        for row, label in zip(scores, labels):
            scaled = [s / t for s in row]
            top = max(scaled)
            lse = top + math.log(sum(math.exp(s - top) for s in scaled))
            nll -= scaled[label] - lse
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t
