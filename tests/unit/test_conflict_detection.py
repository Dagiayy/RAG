from app.services.ingestion.conflict_detection import extract_numeric_claims


def test_extracts_speed_limit_with_unit():
    claims = extract_numeric_claims("Vehicles must not exceed 60 km/h on site roads.")
    assert len(claims) == 1
    assert claims[0].number == 60
    assert claims[0].unit == "km/h"


def test_extracts_currency_claim():
    claims = extract_numeric_claims("The Q1 budget allocation is $420,000 for the department.")
    assert len(claims) == 1
    assert claims[0].number == 420000
    assert claims[0].unit == "dollars"


def test_extracts_multiple_claims_in_one_text():
    claims = extract_numeric_claims(
        "Calibrate every 90 days; batteries last 18 months under normal use."
    )
    units = {c.unit for c in claims}
    numbers = {c.number for c in claims}
    assert "day" in units
    assert "month" in units
    assert 90.0 in numbers
    assert 18.0 in numbers


def test_bare_number_without_unit_is_not_extracted():
    # A page reference / plain count shouldn't be treated as a comparable
    # claim — only NUMBER+UNIT is specific enough to compare safely.
    claims = extract_numeric_claims("See item 42 in the appendix for details.")
    assert claims == []


def test_unit_normalization_strips_trailing_s():
    singular = extract_numeric_claims("Valid for 1 month.")
    plural = extract_numeric_claims("Valid for 3 months.")
    assert singular[0].unit == plural[0].unit == "month"


def test_percent_sign_and_word_both_extracted():
    sign = extract_numeric_claims("Accuracy within plus or minus 2% volumetric water content.")
    word = extract_numeric_claims("Accuracy within plus or minus 2 percent of true value.")
    assert sign[0].unit == "%"
    assert word[0].unit == "percent"
