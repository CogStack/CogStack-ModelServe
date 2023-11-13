from api.auth.backends import get_backends


def test_get_backends():
    backends = get_backends()
    assert len(backends) == 1
    assert backends[0].name == "jwt"
    assert backends[0].transport.scheme.scheme_name == "OAuth2PasswordBearer"
    assert backends[0].get_strategy().algorithm == "HS256"
