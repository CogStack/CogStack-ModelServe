from app.api.auth.schemas import UserRead, UserCreate, UserUpdate
from fastapi_users import schemas


def test_import():
    issubclass(UserRead, schemas.BaseUser)
    issubclass(UserCreate, schemas.BaseUserCreate)
    issubclass(UserUpdate, schemas.BaseUserCreate)
