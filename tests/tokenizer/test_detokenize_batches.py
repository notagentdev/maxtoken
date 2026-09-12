"""Several DetokenizeMsg for one uid in a single batch must decode exactly like
the same messages one at a time.

The per-uid offsets advance once per message, but the batched decode read every
message's slices from the pre-batch state, so a request whose tokens queued up
(speculative commits, or a second concurrent stream) had its text re-appended:
`{"reviewreviewerreviewerId...` (2026-09-12, second of two parallel requests)."""
from maxtoken.message import DetokenizeMsg
from maxtoken.tokenizer.detokenize import DetokenizeManager


class _CharTokenizer:
    """One printable character per id; batch_decode concatenates."""

    eos_token_id = 0

    def batch_decode(self, batches):
        return ["".join(chr(96 + i) for i in ids) for ids in batches]


def _msg(uid, tok, finished=False):
    return DetokenizeMsg(uid=uid, next_token=tok, finished=finished)


def _sequential(msgs):
    m = DetokenizeManager(_CharTokenizer(), frozenset({0}))
    return [m.detokenize([x])[0] for x in msgs]


def test_repeated_uid_in_one_batch_matches_one_at_a_time():
    msgs = [_msg(1, 1), _msg(1, 2), _msg(2, 8), _msg(1, 3), _msg(2, 9), _msg(1, 4)]
    batched = DetokenizeManager(_CharTokenizer(), frozenset({0})).detokenize(msgs)
    assert batched == _sequential(msgs) == ["a", "b", "h", "c", "i", "d"]


def test_text_is_emitted_exactly_once_per_stream():
    msgs = [_msg(7, i) for i in range(1, 12)] + [_msg(7, 0, finished=True)]
    out = DetokenizeManager(_CharTokenizer(), frozenset({0})).detokenize(msgs)
    assert "".join(out) == "abcdefghijk"


def test_single_message_batches_are_unchanged():
    m = DetokenizeManager(_CharTokenizer(), frozenset({0}))
    assert m.detokenize([_msg(3, 5)]) == ["e"]
    assert m.detokenize([_msg(3, 6), _msg(4, 7)]) == ["f", "g"]
