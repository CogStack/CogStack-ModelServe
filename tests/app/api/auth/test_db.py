import pytest
from fastapi_users.db import SQLAlchemyUserDatabase
from app.api.auth.db import make_sure_db_and_tables, get_user_db


@pytest.mark.asyncio
async def test_make_sure_db_and_tables():
    await make_sure_db_and_tables()


@pytest.mark.asyncio
async def test_get_user_db():
    async for user_db in get_user_db():
        assert isinstance(user_db, SQLAlchemyUserDatabase)
