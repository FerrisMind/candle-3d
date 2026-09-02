# `lux3d-server-core`

Core crate that powers the Lux3D HTTP server. It exposes an OpenAI-style async generation API for embedding in external Axum applications.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | Supported models (`pi3`, `pi3x`, `triposr`) |
| `POST` | `/v1/point-clouds/generations` | Create async point-cloud job |
| `POST` | `/v1/meshes/generations` | Create async mesh job |
| `GET` | `/v1/generations` | List recent jobs |
| `GET` | `/v1/generations/{id}` | Poll job status |
| `GET` | `/v1/generations/{id}/content` | Download completed asset |

## Workflow

1. `POST /v1/point-clouds/generations` or `/v1/meshes/generations` with `multipart/form-data`
2. Receive `202 Accepted` with a `generation` job object (`status: queued`)
3. Poll `GET /v1/generations/{id}` until `status: completed`
4. Download the result from `GET /v1/generations/{id}/content`

Run the standalone binary:

```powershell
cargo run -p lux3d-server -- --host 127.0.0.1 --port 8080
```
