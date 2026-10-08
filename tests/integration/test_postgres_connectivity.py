"""Requires `docker compose up -d postgres` to be running locally."""

import pytest
from sqlalchemy import text

from app.repositories.db import get_engine


@pytest.mark.asyncio
async def test_postgres_select_1():
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
