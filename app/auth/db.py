from typing import AsyncGenerator

from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from utils import get_settings


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    pass


_engine = create_async_engine(get_settings().AUTH_DATABASE_URL)
_async_session_maker = async_sessionmaker(_engine, expire_on_commit=False)


async def _get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with _async_session_maker() as session:
        yield session


async def make_sure_db_and_tables() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_user_db(session: AsyncSession = Depends(_get_async_session)) -> AsyncGenerator[SQLAlchemyUserDatabase, None]:
    yield SQLAlchemyUserDatabase(session, User)
