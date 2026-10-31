"""Frozen worst-case gate: replays every book against its stored ref and bound."""

import pytest


def test_worst_book(
    request, test, worst_book, target, accuracy, operation, dtype, worst_key
):
    """Replay one frozen book; a violation aborts the session."""
    maxulp = worst_book.bound(worst_key)
    if maxulp is None:
        pytest.fail("book has no maxulp header")
    xs, ref = worst_book.replay(worst_key)
    if not xs[0].size:
        pytest.skip("empty book")

    try:
        test(
            getattr(target, operation),
            xs,
            ref,
            maxulp,
            accuracy=accuracy,
            # never folds new inputs; re-stamps this book's errors under
            # --worst-collect, and only for a non-reference target
            replay=(worst_book, worst_key),
        )
    except AssertionError:
        # shouldstop is the channel xdist propagates back from a worker; a
        # pytest.exit() would abort one worker and leave the others running.
        request.session.shouldstop = "frozen worst-case regression"
        raise
