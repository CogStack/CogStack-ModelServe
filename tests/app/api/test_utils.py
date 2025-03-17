from fastapi import FastAPI
from app.utils import get_settings
from app.api.utils import (
    add_exception_handlers,
    add_rate_limiter,
    get_rate_limiter,
    encrypt,
    decrypt,
)


def test_add_exception_handlers():
    app = FastAPI()
    add_exception_handlers(app)
    handlers = [handler.__name__ for handler in app.exception_handlers.values()]
    assert "json_decoding_exception_handler" in handlers
    assert "rate_limit_exceeded_handler" in handlers
    assert "start_training_exception_handler" in handlers
    assert "annotation_exception_handler" in handlers
    assert "configuration_exception_handler" in handlers
    assert "unhandled_exception_handler" in handlers


def test_add_middlewares():
    app = FastAPI()
    add_rate_limiter(app, get_settings())
    middlewares = [str(middleware) for middleware in app.user_middleware]
    assert "Middleware(SlowAPIMiddleware)" in middlewares

    streamable_app = FastAPI()
    add_rate_limiter(streamable_app, get_settings(), True)
    middlewares = [str(middleware) for middleware in streamable_app.user_middleware]
    assert "Middleware(SlowAPIASGIMiddleware)" in middlewares


def test_get_per_address_rate_limiter():
    limiter = get_rate_limiter(get_settings(), auth_user_enabled=False)
    assert limiter._key_func.__name__ == "get_remote_address"


def test_get_per_user_rate_limiter():
    limiter = get_rate_limiter(get_settings(), auth_user_enabled=True)
    key_func = limiter._key_func
    assert key_func.__name__ == "_get_user_auth"


def test_encrypt():
    fake_public_key_pem = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ITkTP8Tm/5FygcwY2EQ
7LgVsuCF0OH7psUqvlXnOPNCfX86CobHBiSFjG9o5ZeajPtTXaf1thUodgpJZVZS
qpVTXwGKo8r0COMO87IcwYigkZZgG/WmZgoZART+AA0+JvjFGxflJAxSv7puGlf8
2E+u5Wz2psLBSDO5qrnmaDZTvPh5eX84cocahVVI7X09/kI+sZiKauM69yoy1bdx
16YIIeNm0M9qqS3tTrjouQiJfZ8jUKSZ44Na/81LMVw5O46+5GvwD+OsR43kQ0Te
xMwgtHxQQsiXLWHCDNy2ZzkzukDYRwA3V2lwVjtQN0WjxHg24BTBDBM+v7iQ7cbw
eQIDAQAB
-----END PUBLIC KEY-----"""
    encrypted = encrypt("test", fake_public_key_pem)
    assert isinstance(encrypted, str)
    assert len(encrypted) > 0


def test_decrypt():
    fake_private_key_pem = """-----BEGIN PRIVATE KEY-----
MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDchORM/xOb/kXK
BzBjYRDsuBWy4IXQ4fumxSq+Vec480J9fzoKhscGJIWMb2jll5qM+1Ndp/W2FSh2
CkllVlKqlVNfAYqjyvQI4w7zshzBiKCRlmAb9aZmChkBFP4ADT4m+MUbF+UkDFK/
um4aV/zYT67lbPamwsFIM7mqueZoNlO8+Hl5fzhyhxqFVUjtfT3+Qj6xmIpq4zr3
KjLVt3HXpggh42bQz2qpLe1OuOi5CIl9nyNQpJnjg1r/zUsxXDk7jr7ka/AP46xH
jeRDRN7EzCC0fFBCyJctYcIM3LZnOTO6QNhHADdXaXBWO1A3RaPEeDbgFMEMEz6/
uJDtxvB5AgMBAAECggEABLc80J610yStZmQf90gYng9Tu3cMtYpXoNnfj6Fzp+af
2eIyIg5+zBVU28t4IUzMK86mGj8gxIuQSXHv3uBpNSerWEFGrzkEXfpJFBIPhl3/
HQ3rsT1gGReHMFw8EFE4LoosYOdyaYJv9JSujRarnA6cLWDWp3tLudkNU+bU1A6n
MyXwM1jyM5RkLKSY5tTuzNZ3fL/Yz+Spuxw9yKFE6l6Rcb0weLYMNVrPlSr4SfJ3
R9WyfRKqO2WXZCJ5sGEOx30Zas6ivsorVZ+b9VWkAaDvCpcbg4ahyfGjhWFWFpCo
+zxFlmfGyouY8OtL7Tq7QSnHxoFvMBv7p/CpTuezDwKBgQDrWGjGsAZrD9sIKuYC
yAo7SkaN8s1tm224MYlFd26vzvVxMUv2ZYgRGDPD3L8KDgzIPpU9ltnyPnKmso6c
92+Uit3p1lCLvrRZI+ArYaXkk7pl/XjAd9FNzIWp5mBCOIeEdpeOpBscaOe1yxDG
VvK1RKBqZNX1vkmcjSSRA6So1wKBgQDv32A76d4UQNzjIQeDn/Q4LGZOKPMyC+ys
u/Pf91hGnu6LvcmKjs2HhgOUlH1Nd5voR+bb0AxbdrOV8EtoYoWAg8c5t/jzWspK
UXIRe37EQeKSV6MwU+93Tcjr2fohdGznc6etECa8b9n05qLZa6pt7MtMM2vI69mR
aCGbtnB3LwKBgQDWUeLI3dBae0v6OibQ7Z7zs4ZhCnYtlNfsX6Ak1MjF7fDyrfQB
ZSDugF3TxhlrbLQTP3rlZZUA2AHM8NqS83p3iabhpjwfpwHSE6u3letfJ3EeJCBt
FjBTaydmO9f5NkWjSeRnD+dojdhFY7HZDaFlliOIAGAgtLOQj7B3JxwybQKBgQDc
bwh+xqJhNmJHD5laKmpCHPs/JH6pJTAwZODult02uOM65AQMIsNZoZw0tGiaAiry
QPE0W3KfsuvCBHsnyDIrMe6pahmLeYmg1kvfKQAL1wghuAutY9USbBcSNtSYXeee
ozgZ4FfYn2lKl5BcAYczUYJZ2n9YuvTLnUgVUojz3QKBgQDmewPhaqYJOKDHeY6D
QySZIZwb2mZd3nozPMzBJuTh5QK+KPkzSeJTihuIZh8ZImD0LX3TX8KSdz9oZQQR
cExDsxcGU7ZcTO9WVwDhqF/9ofkXfLOFKxugLNEA5RA3gRcpCxMRLS4k6dfN9N9o
3RQZkF/usTTvyvFQR96frZb2FQ==
-----END PRIVATE KEY-----"""
    encrypted = "TLlMBh4GDf3BSsO/RKlqG5H7Sxv7OXGbl8qE/6YLQPm3coBbnrRRReX7pLamnjLPUU0PtIRIg2H/hWBWE/3cRtXDPT7jMtmGHMIPO/95A0DkrndIkOeQ29J6TBPBBG6YqBNRb2dyhDBwDIEDjPTiRe68sYz4KkxzSOkcz31314kSkZvdIDtQOgeRDa0/7U0VrJePL2N7SJvEiHf4Xa3vW3/20S3O8s/Yp0Azb/kS9dFa54VO1fNNhJ46OtPpdekiFDR5yvQfHwFVeSDdY+eAuYLTWa6bz/LrQkRAdRi9EW5Iz/q8WgKhZXQJfcXtiKfVuFar2N2KodY7C/45vMOfvw=="
    decrypted = decrypt(encrypted, fake_private_key_pem)
    assert decrypted == "test"
