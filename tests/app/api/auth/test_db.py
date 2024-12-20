import pytest
from fastapi_users.db import SQLAlchemyUserDatabase

from api.auth.db import get_user_db, make_sure_db_and_tables


@pytest.mark.asyncio
async def test_make_sure_db_and_tables():
    await make_sure_db_and_tables()


@pytest.mark.asyncio
async def test_get_user_db():
    async for user_db in get_user_db():
        assert isinstance(user_db, SQLAlchemyUserDatabase)
