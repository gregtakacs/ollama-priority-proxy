# Ollama Priority Proxy

An intelligent proxy that sits in front of [Ollama](https://ollama.com/) to manage model priorities and prevent unnecessary model reload churn — especially critical when **multiple users** share a single GPU machine.

## Why This Exists

When multiple users run queries against Ollama on the same GPU, every user's requests trigger model loads and unloads independently. With limited VRAM (e.g., 24 GB), swapping between large models like `TakacsAI-Coder-256k` (~27 GB) and smaller ones like `qwen3:8b` (~14 GB) causes constant **VRAM thrashing**:

- Models get unloaded to free VRAM, only to be reloaded seconds later
- Each reload takes 30–120+ seconds — users experience painful delays
- GPU memory is wasted on repeated load/unload cycles instead of inference
- One user's "heavy" model can displace another user's working model

This proxy solves that by:

1. **Measuring** each model's actual VRAM footprint at a given context length
2. **Assigning priorities** (0–100) to models based on what the team needs most
3. **Intercepting requests** and routing them intelligently:
   - If the requested model fits in available VRAM alongside currently loaded models → use it directly (Ollama handles loading/eviction as needed)
   - If it doesn't fit → rewrite the request to point to the highest-priority model already resident in VRAM instead
4. **Suppressing noise** — no default HTTP logging, only meaningful swap events

## How It Works

```
Client (port 11434 or OpenAI-compatible)
        │
        ▼
┌───────────────────────┐
│   Ollama Priority     │  ← Runs on port 8080
│   Proxy               │
│                       │
│  • Measures VRAM      │
│  • Checks priorities  │
│  • Rewrites requests  │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   Ollama Backend      │  ← port 11434 (or remote)
│                       │
│  • Loads/unloads models│
│  • Runs inference     │
└───────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `OllamaModelProxy.py` | The proxy server — intercepts and routes requests based on VRAM + priority logic |
| `measure_models.py` | Helper script to measure actual VRAM usage for your models |
| `ollama_model_registry.json` | Unified config file with measured VRAM sizes **and** model priorities (sorted by priority) |

## Setup — Step by Step

### 1. Generate the Model Config (`ollama_model_registry.json`)

The proxy needs a JSON config that maps model names to their actual VRAM usage and priority scores. There are three ways to create this:

#### Option A: Measure ALL local models (quickest)

```bash
python3 measure_models.py -a --output ollama_model_registry.json
```

This benchmarks every Ollama model currently installed, skipping ones already measured unless you add `--force`. The default context length is **131072** (128k) tokens. Use `--use-default-ctx` to let each model use its own native default context length instead of forcing a fixed value — recommended for accurate VRAM measurements that match real-world usage via WebUI or CLI.

#### Option B: Measure specific models interactively

```bash
python3 measure_models.py -i --output ollama_model_registry.json
```

You'll be prompted for each model name and context length one at a time.

#### Option C: Batch mode from a config file

Create `batch_config.json`:
```json
{
  "models": [
    {"model": "TakacsAI-Coder-256k:latest", "context": 262144},
    {"model": "qwen3:8b", "context": 8192}
  ]
}
```

Then run:
```bash
python3 measure_models.py --config batch_config.json --output ollama_model_registry.json
```

### 2. Set Model Priorities

Open `ollama_model_registry.json` and edit the `"priority"` values (0–100, higher = more preferred). The file is automatically sorted by priority descending.

Example config:
```json
{
  "models": [
    {"name": "TakacsAI-Coder-256k",  "vram_bytes": 27000000000, "priority": 90},
    {"name": "qwen3:8b",              "vram_bytes": 14000000000, "priority": 50},
    {"name": "qwen2.5-coder:1.5b-base","vram_bytes":  2000000000, "priority": 30}
  ]
}
```

Priority scale reference:
| Priority | Meaning |
|----------|---------|
| **90–100** | Critical — always preferred when resident in VRAM |
| **50–80**  | Important — prefer over lower-priority models |
| **20–40**  | Low priority — fall-back only |
| **1–10**   | Rarely used — last resort |

> **Note:** New measured models default to priority `20`. Unknown models (not in config) get priority `5` and trigger a warning on first use.

### 3. Start the Proxy

#### Running Directly (Python)

```bash
# Set the config file path, then start:
export MODEL_CONFIG_FILE=ollama_model_registry.json
python3 OllamaModelProxy.py
```

The proxy listens on **port 8080** by default and forwards to Ollama at `http://localhost:11434`.

#### Running in Docker (Production)

Build and run with the included Dockerfile. The config file is bind-mounted from an external directory so you can edit priorities without rebuilding:

```bash
# Copy config to a persistent location
mkdir -p $DOCKERDIR/appdata/llm-tools
cp ollama_model_registry.json $DOCKERDIR/appdata/llm-tools/

docker build -t ollama-proxy .
docker run -d \
  --name ollama-proxy \
  --runtime nvidia \
  --network t2_proxy \
  -e OLLAMA_PROXY_TARGET=http://host.docker.internal:11434 \
  -e MODEL_CONFIG_FILE=/app/config/ollama_model_registry.json \
  -p 8080:8080 \
  -v $DOCKERDIR/appdata/llm-tools:/app/config:rw \
  --mount source=ollama_token,target=/run/secrets/ollama_token,type=secret \
  ollama-proxy
```

Key Docker details:
- **`--runtime nvidia`** — injects GPU devices and NVIDIA driver libraries for real VRAM detection
- **Config bind mount** (`$DOCKERDIR/appdata/llm-tools:/app/config`) — edit `ollama_model_registry.json` in place without rebuilding; restart the container to pick up changes
- **Docker secrets** — the proxy reads API keys from files via `_FILE` env vars (e.g., `OLLAMA_PROXY_OLLAMA_API_KEY_FILE=/run/secrets/ollama_token`) for secure credential management

> **Note:** The `measure_models.py` script is not included in the Docker image. Run it on your host (or in a separate container) to generate/update the config file before binding it into the proxy:
> ```bash
> # On host — generates/configures ollama_model_registry.json
> python3 measure_models.py -a --output $DOCKERDIR/appdata/llm-tools/ollama_model_registry.json
> ```

### 4. Point Clients at the Proxy

Replace your Ollama endpoint with the proxy:

- **Ollama API clients:** Use `http://localhost:8080` instead of `http://localhost:11434`
- **OpenAI-compatible clients (e.g., chat UIs):** Set base URL to `http://localhost:8080/v1`
- **Streaming requests** are fully supported — the proxy rewrites model names in SSE events too

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_PROXY_LISTEN_PORT` | `8080` | Port the proxy listens on |
| `OLLAMA_PROXY_TARGET` | `http://localhost:11434` | Target Ollama backend URL |
| `MODEL_CONFIG_FILE` | `ollama_model_registry.json` | Path to unified config JSON (measurements + priorities) |
| `OLLAMA_PROXY_OLLAMA_API_KEY` | *(empty)* | API key for proxy → Ollama calls. Supports `_FILE` suffix for Docker secrets: `OLLAMA_PROXY_OLLAMA_API_KEY_FILE=/run/secrets/ollama_token` |
| `GPU_TOTAL_VRAM_GB` | *(auto-detect)* | Override GPU total VRAM in GB (useful when running headless without nvidia-smi) |

### Using with a locked-down Ollama instance

If your Ollama has authentication enabled, set the proxy's API key:

```bash
# Direct Python — plain env var
export OLLAMA_PROXY_OLLAMA_API_KEY="<your-api-key>"
python3 OllamaModelProxy.py

# Docker — via secrets file (recommended)
docker run -e OLLAMA_PROXY_OLLAMA_API_KEY_FILE=/run/secrets/ollama_token ...
```

This key is used **only** for the proxy's own internal calls to Ollama (e.g., `/api/ps`, `/api/tags`) — these are requests the proxy makes to gather routing information. It is **not** added to client requests forwarded through the proxy.

Forwarded requests (both streaming and non-streaming) pass through whatever `Authorization` header the **client** provides. So if your Ollama requires per-request authentication, clients must include their own API key in their request headers — the proxy does not inject or override it.

## Intercepted Endpoints

The proxy intercepts and rewrites model names on these endpoints:

| Endpoint | Description |
|----------|-------------|
| `POST /api/chat` | Ollama native chat (streaming & non-streaming) |
| `POST /api/generate` | Ollama native generate (streaming & non-streaming) |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |

All other endpoints pass through unchanged.

## License

MIT License — see [LICENSE](LICENSE) for details.
