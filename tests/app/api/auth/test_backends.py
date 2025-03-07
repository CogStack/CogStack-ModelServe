from app.api.auth.backends import get_backends


def test_get_backends():
    backends = get_backends()
    assert len(backends) == 2
    assert backends[0].name == "jwt"
    assert backends[0].transport.scheme.scheme_name == "OAuth2PasswordBearer"
    assert backends[0].get_strategy().algorithm == "HS256"
    assert backends[1].name == "cookie"
    assert backends[1].transport.scheme.scheme_name == "APIKeyCookie"
    assert backends[1].get_strategy().algorithm == "HS256"
