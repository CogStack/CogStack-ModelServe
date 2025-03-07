from fastapi_users.authentication.backend import AuthenticationBackend
from fastapi_users import FastAPIUsers
from app.api.auth.users import Props


def test_props():
    props = Props(True)
    assert isinstance(props.auth_backends[0], AuthenticationBackend)
    assert isinstance(props.fastapi_users, FastAPIUsers)
    assert callable(props.current_active_user)


def test_empty_props():
    props = Props(False)
    assert props.auth_backends == []
    assert props.fastapi_users is None
    assert props.current_active_user() is None
