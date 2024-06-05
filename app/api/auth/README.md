# Authentication Instructions

To enable the authentication features for users, you will need to provide the following environment variables:

* AUTH_USER_ENABLED=true
* AUTH_JWT_SECRET=<JWT_SECRET>
* AUTH_ACCESS_TOKEN_EXPIRE_SECONDS=3600
* AUTH_DATABASE_URL=<DATABASE_URL>

There are several approaches to generating the `<JWT_SECRET>` and one way is by using the `cryptography` package:
```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()
print(key.decode("utf-8"))
```

Your CMS users can be stored either in a local file-based database (e.g., `<DATABASE_URL>` set to `sqlite+aiosqlite:///./cms-users.db` when SQLite is used) or in a remote one (e.g., `<DATABASE_URL>` set to `postgresql+asyncpg://<AUTH_DB_USERNAME>:<AUTH_DB_PASSWORD>@auth-db:5432/cms-users` when you have an [auth-db container](./../../../docker-compose-auth.yml) running).

Currently, user management tasks such as registration and removal are performed by the admin. As an administrator, in order to create a new user, you need to log into the database and create a new record by running:
```sql
cms-users=> INSERT INTO 'user' (id, email, hashed_password, is_active, is_superuser, is_verified) VALUES ('<UUID>', '<EMAIL>', '<HASHED_PASSWORD>', true, false, true)
```

Among the above arguments, `<HASHED_PASSWORD>` can be calculated using the `fastapi_users` package:
```python
from fastapi_users.password import PasswordHelper
helper = PasswordHelper()
print(helper.hash("RAW_PASSWORD"))
```