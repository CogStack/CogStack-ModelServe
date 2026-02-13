# CogStack ModelServe MCP Server

Model Context Protocol (MCP) server for CogStack ModelServe, providing AI assistants with tools to interact with deployed NLP models and their training metrics.

## Quick Start

### 1. Install Dependencies
```bash
pip install '.[mcp]'
```

### 2. Set Environment Variables
```bash
export CMS_BASE_URL="http://127.0.0.1:8000"  # CogStack ModelServe API base URL
export CMS_MCP_API_KEYS="key1,key2,...keyN"  # Optional: The API key(s) for authentication
```

### 3. Run the Server
```bash
# STDIO transport (default)
cms mcp run
```

```bash
# HTTP transport
cms mcp run --transport http
```
Once the above succeeds, the MCP server will be running at http://127.0.0.1:8080/mcp

```bash
# SSE transport
cms mcp run --transport sse
```
Once the above succeeds, the MCP server will be running at http://127.0.0.1:8080/sse

## Claude Desktop Configuration

To use this MCP server with Claude Desktop, add the following configuration to your `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "cms-mcp-server": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://127.0.0.1:8080/sse"
      ]
    }
  }
}
```

With API-key-based authentication:
```bash
cms mcp run --transport sse --cms-mcp-api-keys "key1,key2,...keyN"
```
```json
{
  "mcpServers": {
    "cms-mcp-server": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://127.0.0.1:8080/sse",
        "--header",
        "X-API-Key:${API_KEY_HEADER}"
      ],
      "env": {
        "API_KEY_HEADER": "ONE_OF_THE_API_KEYS"
      }
    }
  }
}
```

With OAuth2-based authentication:
```bash
cms mcp run --transport sse
  --cms-mcp-oauth-enabled \
  --github-client-id <GITHUB_CLIENT_ID> \
  --github-client-secret <GITHUB_CLIENT_SECRET> \
  --google-client-id <GOOGLE_CLIENT_ID> \
  --google-client-secret <GOOGLE_CLIENT_SECRET>
```
```json
{
  "mcpServers": {
    "cms-mcp-server": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://127.0.0.1:8080/sse",
        "--header",
        "X-API-Key:${AUTH_HEADER}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <ACCESS_TOKEN>"
      }
    }
  }
}
```

## Available Tools

| Tool | Description | Arguments |
|------|-------------|-----------|
| `get_model_info` | Get running model information | None |
| `get_annotations` | Extract entities from text | `text: str` |
| `redact_text` | Redact sensitive information | `text: str` |
| `get_train_eval_info` | Get training/evaluation status | `train_eval_id: str` |
| `get_train_eval_metrics` | Get training/evaluation metrics | `train_eval_id: str` |

## Configuration

| Environment Variable | Default                 | Description |
|---------------------|-------------------------|-------------|
| `CMS_BASE_URL` | `http://127.0.0.1:8000` | ModelServe API base URL |
| `CMS_MCP_SERVER_HOST` | `127.0.0.1` | MCP server host |
| `CMS_MCP_SERVER_PORT` | `8080` | MCP server port |
| `CMS_MCP_TRANSPORT` | `stdio` | Transport type (`stdio`, `http` or `sse`) |
| `CMS_ACCESS_TOKEN` | Empty | Bearer token for ModelServe API |
| `CMS_API_KEY` | `Bearer` | API key for ModelServe API |
| `CMS_MCP_API_KEYS` | None | Comma-separated API keys for authentication |
| `CMS_MCP_OAUTH_ENABLED` | `false` | Enable OAuth authentication |
| `CMS_MCP_BASE_URL` | `http://<host>:<port>`  | Base URL for OAuth callback |
| `CMS_MCP_DEV` | `0` | Run in development mode |


## Authentication

The server supports two authentication methods:

### 1. API Key Authentication
When `CMS_MCP_API_KEYS` is set, clients must authenticate using:
- **Header**: `X-API-Key: your-key`

### 2. OAuth Authentication (SSE Transport)
When `CMS_MCP_OAUTH_ENABLED=true`, the server provides a built-in OAuth 2.0 login flow for SSE transport.

**OAuth Endpoints:**
- `/oauth/login` - Login page with Google and GitHub options
- `/oauth/authorize/{provider}` - Initiates OAuth flow for the specified provider
- `/oauth/callback/{provider}` - OAuth callback handler
- `/oauth/status` - Check current session status
- `/oauth/logout` - Logout and clear session

**Environment Variables for OAuth:**
| Variable | Description |
|----------|-------------|
| `GITHUB_CLIENT_ID` | GitHub OAuth client ID |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth client secret |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |

**Note:** If OAuth credentials are not configured, the server will log a warning but continue running.

**Session Authentication:**
After OAuth login, a session cookie (`cms_mcp_session`) is set. Subsequent MCP requests should include this cookie for authentication.
