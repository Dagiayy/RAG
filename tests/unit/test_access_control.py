from app.core.security.access import authorized_access_levels, can_access
from app.models.document import AccessLevel
from app.models.user import ClearanceLevel


def test_public_clearance_only_sees_public():
    assert authorized_access_levels(ClearanceLevel.PUBLIC) == ["public"]


def test_restricted_clearance_sees_everything():
    levels = authorized_access_levels(ClearanceLevel.RESTRICTED)
    assert set(levels) == {"public", "internal", "department", "confidential", "restricted"}


def test_department_clearance_sees_up_to_department_not_confidential():
    levels = authorized_access_levels(ClearanceLevel.DEPARTMENT)
    assert set(levels) == {"public", "internal", "department"}
    assert "confidential" not in levels
    assert "restricted" not in levels


def test_can_access_true_when_clearance_meets_or_exceeds():
    assert can_access(ClearanceLevel.RESTRICTED, AccessLevel.CONFIDENTIAL) is True
    assert can_access(ClearanceLevel.CONFIDENTIAL, AccessLevel.CONFIDENTIAL) is True


def test_can_access_false_when_clearance_below_requirement():
    assert can_access(ClearanceLevel.PUBLIC, AccessLevel.CONFIDENTIAL) is False
    assert can_access(ClearanceLevel.DEPARTMENT, AccessLevel.RESTRICTED) is False
