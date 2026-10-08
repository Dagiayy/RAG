"""AuthProvider protocol (docs/security.md): callers depend on this
interface, not a concrete implementation, so a later production stage can
swap in real OIDC/SSO without touching endpoint code (docs/deployment.md
stage 2). `StaticUserAuthProvider` is the local/dev implementation — a
Postgres `users` table looked up by bearer token (`api_key`).
"""

from typing import Protocol

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.db import get_db_session

_bearer_scheme = HTTPBearer(auto_error=False)


class AuthProvider(Protocol):
    async def authenticate(self, session: AsyncSession, token: str) -> User | None: ...


class StaticUserAuthProvider:
    async def authenticate(self, session: AsyncSession, token: str) -> User | None:
        return await session.scalar(select(User).where(User.api_key == token))


_auth_provider: AuthProvider = StaticUserAuthProvider()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await _auth_provider.authenticate(session, credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unknown API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
