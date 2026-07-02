#!/usr/bin/env python3
"""
Ollama Smart Proxy v7 — Priority-Based Model Routing

Proxy on port 8080 intercepts Ollama API requests and resolves model names
based on a unified config file containing VRAM measurements AND priority scores.

Routing logic:
  1. If the requested model fits in VRAM alongside currently loaded models → use it directly.
  2. If it doesn't fit → fall back to the highest-priority model already resident in VRAM.
  3. If nothing is loaded → pass through unchanged (Ollama loads its default).

Config file format (unified: measurements + priorities):
    {
      "models": [
        {"name": "TakacsAI-Coder-256k",  "vram_bytes": 27000000000, "priority": 90},
        {"name": "qwen3:8b",              "vram_bytes": 14000000000, "priority": 50},
        {"name": "qwen2.5-coder:1.5b-base","vram_bytes":  2000000000, "priority": 30}
      ]
    }

Intercepted endpoints (model rewriting):
  - POST /api/chat          (with or without ?stream=true)
  - POST /api/generate      (with or without ?stream=true)
  - POST /v1/chat/completions (OpenAI-compatible, with or without stream)

All other endpoints pass through unchanged.

Environment Variables:
  OLLAMA_PROXY_LISTEN_PORT     Default listening port (default: 8080)
  OLLAMA_PROXY_TARGET          Target Ollama host (default: http://localhost:11434)
  MODEL_CONFIG_FILE            Path to unified config JSON file (required for routing)
  OLLAMA_PROXY_OLLAMA_API_KEY  API key for proxy → Ollama backend calls (if auth enabled)
  GPU_TOTAL_VRAM_GB            Override GPU total VRAM in GB (default: auto-detect from nvidia-smi)

Priority Scale (0-100, higher = more preferred):
  - Unknown model (not in config):     5   — last resort only; logs a warning on first use
  - Newly measured model (before edit): 20  — low priority until you bump it up
  - User-defined priorities:            1-100 — tune to your preference

Setup:
  1. Measure models with the helper script:
       python measure_models.py -a --output measured_vram.json
  2. Edit measured_vram.json to set desired priorities (sorted by priority automatically)
  3. Start proxy:
       export MODEL_CONFIG_FILE=measured_vram.json
       python OllamaModelProxy.py
"""

import http.server
import json
import os
import subprocess
from urllib import request as urlrequest
from urllib.error import URLError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LISTEN_PORT = int(os.environ.get("OLLAMA_PROXY_LISTEN_PORT", "8080"))
TARGET_HOST = os.environ.get("OLLAMA_PROXY_TARGET", "http://localhost:11434").rstrip("/")

# Path to unified config file (required for priority-based routing)
MODEL_CONFIG_FILE = os.environ.get("MODEL_CONFIG_FILE", "ollama_model_registry.json")

# API key for proxy → Ollama backend calls (if Ollama has auth enabled)
OLLAMA_API_KEY = os.environ.get("OLLAMA_PROXY_OLLAMA_API_KEY", "")

# Default priority for unknown models not in config
UNKNOWN_MODEL_PRIORITY = 5


# ---------------------------------------------------------------------------
# GPU / VRAM Helpers
# ---------------------------------------------------------------------------

_gpu_total_bytes = int(os.environ.get("GPU_TOTAL_VRAM_GB", "0")) * (1024 ** 3) if os.environ.get("GPU_TOTAL_VRAM_GB") else None


def format_bytes(b):
    """Format bytes as human-readable string."""
    if b is None or b == 0:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(b) < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"


def get_gpu_baseline():
    """Get VRAM used by non-Ollama processes (bytes). Returns None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None

        total = 0.0
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                name = parts[1].lower()
                used_mb = float(parts[2]) * 1024 * 1024
                if "ollama" not in name and "nvprocess" not in name:
                    total += used_mb
            except (ValueError, IndexError):
                continue

        return int(total)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_gpu_total_bytes():
    """Get actual GPU total VRAM from nvidia-smi. Falls back to env override or 32 GB default."""
    global _gpu_total_bytes
    if _gpu_total_bytes:
        return _gpu_total_bytes

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            total_mb = float(lines[0])
            _gpu_total_bytes = int(total_mb * 1024 * 1024)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not _gpu_total_bytes:
        print("[proxy] WARNING: Could not detect GPU total VRAM. Using default 32 GB.")
        _gpu_total_bytes = 32 * (1024 ** 3)

    return _gpu_total_bytes


def get_real_available_vram(loaded_ollama_bytes=None):
    """Get real available VRAM for Ollama models.

    Returns: total GPU memory - non-Ollama baseline - currently loaded Ollama usage.
    """
    gpu_total = get_gpu_total_bytes()
    baseline = get_gpu_baseline() or 0
    ollama_used = loaded_ollama_bytes or 0
    available = gpu_total - baseline - ollama_used

    # Debug logging to trace VRAM calculation
    print(f"[vram] total={gpu_total/(1024**3):.2f} GB, baseline={baseline/(1024**3):.2f} GB, "
          f"ollama_used={ollama_used/(1024**3):.2f} GB, available={available/(1024**3):.2f} GB")

    return available


def print_gpu_status(loaded_ollama_bytes=None):
    """Print GPU status summary at startup."""
    gpu_total = get_gpu_total_bytes()
    baseline = get_gpu_baseline()

    if baseline is None:
        print("[proxy] WARNING: Could not detect non-Ollama VRAM usage. Assuming 0.")
        baseline = 0

    ollama_used = loaded_ollama_bytes or 0
    real_available = gpu_total - baseline - ollama_used

    print(f"  GPU Total:       {format_bytes(gpu_total)}")
    if baseline > 0:
        print(f"  Other Processes: {format_bytes(baseline)}")
    print(f"  Ollama Models:   {format_bytes(ollama_used)}")
    print(f"  Real Available:  {format_bytes(real_available)}")


# ---------------------------------------------------------------------------
# Unified Config Loading
# ---------------------------------------------------------------------------

MODEL_CONFIG = {"models": []}  # loaded at startup from MODEL_CONFIG_FILE


def load_model_config():
    """Load unified config file (measurements + priorities)."""
    global MODEL_CONFIG
    if not MODEL_CONFIG_FILE or not os.path.exists(MODEL_CONFIG_FILE):
        print(f"[proxy] No model config file at '{MODEL_CONFIG_FILE}' — using estimates only.")
        return

    try:
        with open(MODEL_CONFIG_FILE) as f:
            data = json.load(f)
        if "models" not in data:
            print(f"[proxy] WARNING: Config missing 'models' key — treating as empty.")
            MODEL_CONFIG = {"models": []}
            return

        MODEL_CONFIG = data
        # Sort by priority DESC, then vram_bytes DESC (file should already be sorted)
        MODEL_CONFIG["models"].sort(
            key=lambda m: (m.get("priority", 0), m.get("vram_bytes", 0)),
            reverse=True
        )

        print(f"[proxy] Loaded {len(MODEL_CONFIG['models'])} model(s) from '{MODEL_CONFIG_FILE}':")
        for m in MODEL_CONFIG["models"]:
            pri = m.get("priority", "?")
            name = m["name"]
            vram = format_bytes(m.get("vram_bytes", 0))
            print(f"    [{pri:3d}] {name}: {vram}")

    except (json.JSONDecodeError, OSError) as e:
        print(f"[proxy] WARNING: Failed to load model config: {e}")


# ---------------------------------------------------------------------------
# Model Resolution Helpers
# ---------------------------------------------------------------------------

def normalize_model_name(name):
    """Strip tag suffix from a model name for config lookup.

    E.g., 'TakacsAI-Coder-256k:latest' -> 'takacsai-coder-256k',
          'qwen3:8b' -> 'qwen3'.
    """
    if not name:
        return ""
    # Strip tag (everything after ':') and lowercase for comparison
    base = name.split(":")[0] if ":" in name else name
    return base.lower()


def get_model_from_config(model_name):
    """Look up a model entry in the config by name. Returns dict or None.

    Strips tags before comparison so 'TakacsAI-Coder-256k:latest' matches
    config entry 'takacsai-coder-256k'.
    """
    norm = normalize_model_name(model_name)
    if not norm:
        return None
    for m in MODEL_CONFIG.get("models", []):
        if normalize_model_name(m["name"]) == norm:
            return m
    return None


def priority_of(model_name):
    """Get the priority for a model name from config. Falls back to default for unknowns."""
    entry = get_model_from_config(model_name)
    if entry:
        return entry.get("priority", 0)

    # Unknown model — log warning and use default priority
    print(f"[proxy] WARNING: '{model_name}' not in config — using default priority {UNKNOWN_MODEL_PRIORITY}")
    return UNKNOWN_MODEL_PRIORITY


def get_vram_for_model(model_name):
    """Get VRAM size and context length for a model from config. Returns (vram_bytes, ctx_len) tuple."""
    entry = get_model_from_config(model_name)
    if entry:
        vram = entry.get("vram_bytes", 0)
        ctx = entry.get("ctx_len")
        return vram, ctx
    # Unknown — estimate conservatively (~14 GB as fallback), no context info
    print(f"[proxy] No measurement for '{model_name}' — assuming ~14 GB.")
    return 14e9, None


# ---------------------------------------------------------------------------
# Authenticated Ollama API Calls
# ---------------------------------------------------------------------------

def _make_request(url, data=None):
    """Make a request to the Ollama backend with optional auth header."""
    req = urlrequest.Request(url, data=data)
    if OLLAMA_API_KEY:
        req.add_header("Authorization", f"Bearer {OLLAMA_API_KEY}")

    try:
        resp = urlrequest.urlopen(req, timeout=10)
        return json.loads(resp.read())
    except (URLError, OSError, ValueError):
        return None


def get_loaded_models():
    """Return list of model dicts sorted by VRAM size (largest first).

    Each dict has: name, size_vram (exact bytes incl. KV cache), digest.
    Returns empty list on failure with a warning.
    """
    data = _make_request(f"{TARGET_HOST}/api/ps")
    if not data or "models" not in data:
        print("[proxy] WARNING: /api/ps failed — model resolution disabled.")
        return []

    models = [{
        "name": m["name"],
        "size_vram": m.get("size_vram", 0),   # exact bytes incl. KV cache
        "digest": m.get("digest", ""),
    } for m in data.get("models", [])]

    return sorted(models, key=lambda x: x["size_vram"], reverse=True)


# ---------------------------------------------------------------------------
# VRAM Fit Check
# ---------------------------------------------------------------------------

def would_fit(model_name, loaded_models):
    """Check if the requested model would fit alongside currently loaded models.

    Uses pre-measured data from config first; falls back to conservative estimate.
    Accounts for non-Ollama GPU processes via real available VRAM calculation.

    The key insight: get_real_available_vram() already subtracts baseline AND loaded Ollama usage,
    so we only need to check if 'measured' fits in the remaining space — no double subtraction needed.
    """
    ollama_used = sum(m["size_vram"] for m in loaded_models)
    total_available = get_real_available_vram(ollama_used)  # real free VRAM (gpu_total - baseline - ollama_used)

    measured, ctx_measured = get_vram_for_model(model_name)
    if measured > 0:
        fits = total_available >= measured
        remaining = total_available - measured
        ctx_info = f" (ctx={ctx_measured})" if ctx_measured else ""
        if fits:
            print(f"[fit] '{model_name}' ({measured/1e9:.2f} GB) fits — {remaining/1e9:.2f} GB free after.{ctx_info}")
        else:
            needed = measured - total_available
            print(f"[fit] '{model_name}' ({measured/1e9:.2f} GB) does NOT fit — would need {needed/1e9:.2f} GB more. "
                  f"({format_bytes(remaining)} free of {format_bytes(total_available)} total).{ctx_info}")
        return fits

    # Shouldn't reach here since get_vram_for_model returns 0 only for truly unknown models,
    # but we handle it as a fallback estimate anyway.
    print(f"[fit] No measurement for '{model_name}' — assuming ~14 GB (conservative).")
    est_size = 14e9
    fits = total_available >= est_size
    return fits


# ---------------------------------------------------------------------------
# Core Routing Logic (Priority-Based)
# ---------------------------------------------------------------------------

def resolve_model(requested, loaded_models):
    """Resolve a requested model name to an actual resident or loadable model.

    Algorithm:
      0. If the requested model is already resident in VRAM → use it directly (no swap needed).
      1. If the requested model would fit in VRAM alongside currently loaded models → use it directly.
         This lets Ollama load/swap to the requested model (it handles eviction if needed).
      2. If it doesn't fit → fall back to the highest-priority model already resident in VRAM.
         The proxy will rewrite the request body to point to that resident model instead.
      3. If nothing is loaded at all → pass through unchanged (Ollama loads its default).

    Returns the resolved model name (or alias) to use in the request body.
    """
    if not requested:
        return None  # no model specified — leave alone

    # Step 0: Check if requested model is already resident in VRAM
    norm_requested = normalize_model_name(requested)
    for m in loaded_models:
        if normalize_model_name(m["name"]) == norm_requested:
            print(f"[route] '{requested}' → use directly (already loaded, {m['size_vram']/(1024**3):.2f} GB).")
            return requested

    # Step 1: Check if requested model fits alongside currently loaded models
    if would_fit(requested, loaded_models):
        print(f"[route] '{requested}' → use directly (fits in VRAM).")
        return requested

    # Step 2: Doesn't fit — find highest-priority resident model to fall back on
    if not loaded_models:
        print("[route] Nothing is loaded. Pass-through unchanged; Ollama will load its default.")
        return requested  # pass through; Ollama handles loading

    # Sort loaded models by priority DESC, then VRAM size DESC (tiebreaker)
    best = max(loaded_models, key=lambda m: (priority_of(m["name"]), m["size_vram"]))
    pri = priority_of(best["name"])
    vram_gb = best["size_vram"] / (1024 ** 3)

    print(f"[route] '{requested}' doesn't fit. "
          f"Falling back to highest-priority resident: '{best['name']}' (pri={pri}, {vram_gb:.2f} GB).")
    return best["name"]


# ---------------------------------------------------------------------------
# Body Rewriting Helpers
# ---------------------------------------------------------------------------

def rewrite_json_body(body_bytes):
    """Parse JSON body, resolve model name via priority routing, return rewritten bytes."""
    try:
        payload = json.loads(body_bytes)
    except (json.JSONDecodeError, ValueError):
        return body_bytes  # can't parse — pass through unchanged

    loaded = get_loaded_models()
    requested = payload.get("model", "")
    final_model = resolve_model(requested, loaded)

    if final_model and final_model != requested:
        payload["model"] = final_model
        print(f"[rewrite] '{requested}' -> '{final_model}'")
        return json.dumps(payload).encode()

    return body_bytes


def rewrite_streaming_body(body_bytes):
    """Rewrite model name in SSE streaming response (WebUI uses this)."""
    try:
        lines = body_bytes.decode().strip().split("\n")
        if not lines or not lines[0].startswith("data: "):
            return body_bytes  # not SSE — treat as regular JSON

        first_line = json.loads(lines[0][6:])
        requested = first_line.get("model", "")
    except (json.JSONDecodeError, IndexError):
        return body_bytes

    loaded = get_loaded_models()
    final_model = resolve_model(requested, loaded)

    if final_model and final_model != requested:
        rewritten_lines = []
        for line in lines:
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    chunk["model"] = final_model
                    rewritten_lines.append(f"data: {json.dumps(chunk)}")
                except (json.JSONDecodeError, ValueError):
                    rewritten_lines.append(line)
            else:
                rewritten_lines.append(line)
        return ("\n".join(rewritten_lines)).encode()

    return body_bytes


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class SmartProxyHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress default logging — only print swap events via print()

    def _forward_raw(self, method, path, headers, body_bytes=None):
        """Forward request to Ollama backend. Returns (response_body, response)."""
        url = f"{TARGET_HOST}{path}"
        req = urlrequest.Request(url, data=body_bytes, method=method)

        # Pass through all client headers unchanged — including Authorization.
        # The proxy's own API key is only used for internal Ollama API calls (e.g., /api/ps).
        for k, v in headers.items():
            req.add_header(k, v)

        try:
            resp = urlrequest.urlopen(req, timeout=300)
            return resp.read(), resp
        except URLError as e:
            # Pass through the actual HTTP error status and body from Ollama.
            # For 4xx/5xx responses, the response body contains useful error info.
            if hasattr(e, 'code') and hasattr(e, 'read'):
                try:
                    resp_body = e.read()
                    code = e.code
                except Exception:
                    resp_body = b""
                    code = 502
                self._send_error(code, f"Ollama error: {resp_body.decode('utf-8', errors='replace')}")
            else:
                self._send_error(502, f"Ollama unreachable: {e}")
            return b"", None

    def _forward_streaming(self, method, path, headers):
        """Forward request and stream SSE response back to client (no buffering)."""
        url = f"{TARGET_HOST}{path}"
        req = urlrequest.Request(url, data=None if self.command == "GET" else b"", method=method)
        for k, v in headers.items():
            req.add_header(k, v)

        try:
            resp = urlrequest.urlopen(req, timeout=300)
        except URLError as e:
            if hasattr(e, 'code') and hasattr(e, 'read'):
                code = e.code
                try:
                    resp_body = e.read().decode('utf-8', errors='replace')
                    self._send_error(code, f"Ollama error: {resp_body}")
                except Exception:
                    self._send_error(502, f"Ollama unreachable: {e}")
            else:
                self._send_error(502, f"Ollama unreachable: {e}")
            return

        # Stream SSE events back to client without buffering
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() != "transfer-encoding":
                self.send_header(k, v)
        self.end_headers()

        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except BrokenPipeError:
                break

    def _forward_json(self, method, path, headers, body_bytes):
        """Forward request and return JSON response (for non-streaming)."""
        resp_body, resp = self._forward_raw(method, path, headers, body_bytes)
        if resp is None:
            return

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding",):
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(resp_body)

    def _handle(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        method = self.command.upper()

        # --- ENDPOINTS TO INTERCEPT (model rewriting on all models) ---
        if method == "POST" and (
            "/api/chat" in path or
            "/api/generate" in path or
            "/v1/chat/completions" in path
        ):
            is_streaming = "?stream=true" in self.path.lower() or "&stream=true" in self.path.lower()

            if is_streaming:
                length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(length) if length else b""
                rewritten = rewrite_streaming_body(body_bytes)

                headers = {k: v for k, v in self.headers.items() if k.lower() != "content-length"}
                headers["Content-Length"] = str(len(rewritten))
                self._forward_streaming("POST", path, headers)
            else:
                length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(length) if length else b""
                rewritten = rewrite_json_body(body_bytes)

                headers = {k: v for k, v in self.headers.items() if k.lower() != "content-length"}
                headers["Content-Length"] = str(len(rewritten))
                self._forward_json("POST", path, headers, rewritten)
            return

        # --- ALL OTHER ENDPOINTS (pass through unchanged) ---
        length = int(self.headers.get("Content-Length", 0)) if method == "POST" else 0
        body_bytes = self.rfile.read(length) if length else b""
        headers = {k: v for k, v in self.headers.items() if k.lower() != "content-length"}
        self._forward_json(method, path, headers, body_bytes)

    def _send_error(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())

    # HTTP verb handlers
    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()


def main():
    print(f"Ollama Smart Proxy v7 starting...")
    print(f"  Listening on     : http://0.0.0.0:{LISTEN_PORT}")
    print(f"  Target Ollama    : {TARGET_HOST}")

    if MODEL_CONFIG_FILE:
        print(f"  Config file      : '{MODEL_CONFIG_FILE}'")
    else:
        print(f"  Config file      : (none — using estimates only)")

    if OLLAMA_API_KEY:
        print(f"  Auth enabled     : yes (proxy → Ollama backend)")
    else:
        print(f"  Auth enabled     : no")

    load_model_config()

    # Print GPU status summary at startup
    print_gpu_status()
    print()
    server = http.server.HTTPServer(("0.0.0.0", LISTEN_PORT), SmartProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()