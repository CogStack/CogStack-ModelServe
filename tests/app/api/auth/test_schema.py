from fastapi_users import schemas

from api.auth.schemas import UserCreate, UserRead, UserUpdate


def test_import():
    issubclass(UserRead, schemas.BaseUser)
    issubclass(UserCreate, schemas.BaseUserCreate)
    issubclass(UserUpdate, schemas.BaseUserCreate)
