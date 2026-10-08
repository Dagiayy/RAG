"""Seeds demo users at each clearance level for testing access control.
Prints each user's bearer token — these are dev-only credentials for a
local synthetic dataset, never real credentials.

Run: python scripts/seed_users.py
Idempotent: re-running prints existing users' tokens rather than
regenerating them.
"""

import asyncio

from sqlalchemy import select

from app.models.user import ClearanceLevel, User, UserRole
from app.repositories.db import get_session_factory

DEMO_USERS = [
    ("admin_user", UserRole.ADMIN, ClearanceLevel.RESTRICTED, None),
    ("manager_user", UserRole.MANAGER, ClearanceLevel.CONFIDENTIAL, "Field Operations"),
    ("employee_user", UserRole.EMPLOYEE, ClearanceLevel.DEPARTMENT, "Data Analysis & Analytics"),
    ("public_viewer", UserRole.VIEWER, ClearanceLevel.PUBLIC, None),
]


async def main() -> None:
    factory = get_session_factory()
    async with factory() as session:
        for username, role, clearance, department in DEMO_USERS:
            existing = await session.scalar(select(User).where(User.username == username))
            if existing:
                user = existing
            else:
                user = User(
                    username=username, role=role, clearance_level=clearance, department=department
                )
                session.add(user)
                await session.flush()
            label = f"{username:16s} role={role.value:10s} clearance={clearance.value:12s}"
            print(f"{label} token={user.api_key}")
        await session.commit()


if __name__ == "__main__":
    asyncio.run(main())
