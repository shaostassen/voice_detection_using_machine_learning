"""Alignment backtrace and per-word correctness labels.

These labels are the ground truth for every confidence experiment, so the
tests check hand-worked cases rather than just exercising the code path.
"""
import pytest

from speechlens.metrics import (DEL, EQUAL, INS, SUB, align, labels_for_words,
                                levenshtein, word_labels)


def test_identical_sequences_are_all_equal():
    ops = align(["a", "b", "c"], ["a", "b", "c"])
    assert [o[0] for o in ops] == [EQUAL, EQUAL, EQUAL]
    assert [o[2] for o in ops] == [0, 1, 2]


def test_substitution_is_not_spelled_as_insert_plus_delete():
    # The diagonal must win ties, otherwise one error would consume two slots
    # and every downstream per-word label would be shifted.
    ops = align(["a", "b", "c"], ["a", "x", "c"])
    assert [o[0] for o in ops] == [EQUAL, SUB, EQUAL]


def test_deletion_carries_no_hypothesis_index():
    ops = align(["a", "b", "c"], ["a", "c"])
    dels = [o for o in ops if o[0] == DEL]
    assert len(dels) == 1
    assert dels[0][1] == 1        # ref index of the dropped "b"
    assert dels[0][2] is None     # nothing on the hypothesis side


def test_insertion_carries_no_reference_index():
    ops = align(["a", "c"], ["a", "b", "c"])
    ins = [o for o in ops if o[0] == INS]
    assert len(ins) == 1
    assert ins[0][1] is None
    assert ins[0][2] == 1


def test_every_hypothesis_token_appears_exactly_once():
    # The property that makes per-word labelling well defined.
    ref = "the quick brown fox jumps".split()
    hyp = "the quik brown cat leaps over".split()
    ops = align(ref, hyp)
    hyp_indices = sorted(o[2] for o in ops if o[2] is not None)
    assert hyp_indices == list(range(len(hyp)))


def test_alignment_cost_matches_levenshtein():
    ref = "a b c d e".split()
    hyp = "a x c e f".split()
    ops = align(ref, hyp)
    cost = sum(1 for o in ops if o[0] != EQUAL)
    assert cost == levenshtein(ref, hyp)


@pytest.mark.parametrize("ref,hyp", [
    ("", "hello world"),
    ("hello world", ""),
    ("", ""),
])
def test_empty_sides(ref, hyp):
    r, h = ref.split(), hyp.split()
    ops = align(r, h)
    assert sum(1 for o in ops if o[2] is not None) == len(h)
    assert sum(1 for o in ops if o[1] is not None) == len(r)


def test_word_labels_line_up_with_hypothesis_words():
    ref = "the quick brown fox"
    hyp = "the quick brown dog"
    labels = word_labels(ref, hyp)
    assert labels == [True, True, True, False]


def test_word_labels_mark_insertions_wrong():
    labels = word_labels("the fox", "the red fox")
    assert len(labels) == 3
    assert labels.count(False) == 1


def test_word_labels_ignore_case_and_punctuation():
    # normalize_text runs first, so LibriSpeech-style uppercase unpunctuated
    # references compare fairly against Whisper's cased, punctuated output.
    labels = word_labels("THE QUICK BROWN FOX", "The quick, brown fox.")
    assert labels == [True, True, True, True]


def test_deletions_contribute_no_label():
    # A dropped word is invisible to any per-word confidence signal: there is
    # no hypothesis token to attach a score to. Documents the blind spot.
    labels = word_labels("the quick brown fox", "the brown fox")
    assert labels == [True, True, True]


# --- labels_for_words: one label per ASR word, whatever normalization does ---

def test_labels_for_words_returns_one_label_per_word():
    # The invariant that keeps confidence scores paired with their labels.
    words = [" The", " quick", " brown", " fox."]
    labels = labels_for_words("the quick brown fox", words)
    assert len(labels) == len(words)
    assert labels == [True, True, True, True]


def test_multi_token_word_is_correct_only_if_all_tokens_align():
    # "don't" normalizes to two tokens; both must match. This is the case that
    # silently desynchronized scores from labels before.
    words = ["I", "don't", "know"]
    assert labels_for_words("i don't know", words) == [True, True, True]
    assert labels_for_words("i do not care", words)[2] is False


def test_word_that_normalizes_to_nothing_is_none_not_false():
    # Bare punctuation carries no evidence; guessing False would poison the
    # error class with tokens that were never wrong.
    words = ["hello", "--", "world"]
    labels = labels_for_words("hello world", words)
    assert labels[1] is None
    assert labels[0] is True and labels[2] is True


def test_labels_track_the_right_word_when_counts_differ():
    # Hypothesis has an extra word; the error must land on the inserted one,
    # not shift every later label.
    words = ["the", "big", "red", "fox"]
    labels = labels_for_words("the red fox", words)
    assert len(labels) == 4
    assert labels[0] is True and labels[3] is True
    assert labels.count(False) == 1
