import pytest

from app.services.evaluation.calibration import calibrate, weighted_cohens_kappa


def test_perfect_agreement_kappa_is_1():
    assert weighted_cohens_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_kappa_degenerate_both_raters_constant_and_equal_is_1():
    # No variance in either rater, and they agree -> trivially perfect
    assert weighted_cohens_kappa([3, 3, 3], [3, 3, 3]) == pytest.approx(1.0)


def test_kappa_no_variance_within_raters_matches_chance_baseline():
    # Judge always says 1, human always says 5: expected agreement under
    # independence is also always a mismatch, so weighted kappa is 0 (no
    # better or worse than what the degenerate marginals alone predict).
    assert weighted_cohens_kappa([1, 1, 1], [5, 5, 5]) == pytest.approx(0.0)


def test_kappa_small_disagreements_score_higher_than_large_ones():
    # Same judge scores and the same human marginal distribution in both
    # cases (just a permutation of 1-5), isolating the effect of gap size:
    # off-by-one mismatches vs. near-reversed (mostly far-apart) mismatches.
    judge = [1, 2, 3, 4, 5]
    small_gap = weighted_cohens_kappa(judge, [2, 1, 4, 3, 5])
    large_gap = weighted_cohens_kappa(judge, [5, 4, 1, 2, 3])
    assert small_gap > large_gap


def test_kappa_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        weighted_cohens_kappa([1, 2], [1])


def test_kappa_empty_raises():
    with pytest.raises(ValueError):
        weighted_cohens_kappa([], [])


def test_calibrate_reports_n_and_agreement_rate():
    result = calibrate([5, 4, 3], [5, 4, 2])
    assert result.n == 3
    assert result.agreement_rate == pytest.approx(2 / 3)
    assert -1.0 <= result.weighted_kappa <= 1.0


def test_calibrate_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        calibrate([1, 2], [1])
