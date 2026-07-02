#!/usr/bin/env python3
"""
measure_models.py — Measure VRAM usage for Ollama models at their native context lengths.

This tool queries each model via /api/show to get its designed context_length,
then measures the exact total VRAM (including KV cache) at that context. If a
model can't fit at its native ctx, it falls back to successively smaller contexts
until one works.

The output is a unified JSON file containing measurements AND priority scores
used by the proxy for routing.

Unified config format:
    {
      "models": [
        {"name": "TakacsAI-Coder-256k",  "vram_bytes": 27000000000, "ctx_len": 262144, "priority": 90},
        {"name": "qwen3:8b",              "vram_bytes": 14000000000, "ctx_len": 32768,  "priority": 50},
        {"name": "qwen2.5-coder:1.5b-base","vram_bytes":  2000000000, "ctx_len": 4096,   "priority": 30}
      ]
    }

Entries are stored sorted by priority DESC (highest first).

Usage:
    # Benchmark ALL local Ollama models at their native context lengths:
    python measure_models.py -a --output measured_vram.json

    # Force recalculation of all models:
    python measure_models.py -a --force --output measured_vram.json

    # Remove stale entries for deleted models:
    python measure_models.py --cleanup --output measured_vram.json

Environment Variables:
    OLLAMA_PROXY_TARGET       Target Ollama host (default: http://localhost:11434)
    OLLAMA_PROXY_OLLAMA_API_KEY  API key for backend calls (if auth enabled)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from urllib import request as urlrequest
from urllib.error import URLError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_HOST = os.environ.get("OLLAMA_PROXY_TARGET", "http://localhost:11434").rstrip("/")
OLLAMA_API_KEY = os.environ.get("OLLAMA_PROXY_OLLAMA_API_KEY", "")

# Default priorities (0-100 scale, like Linux nice — higher = more preferred)
DEFAULT_PRIORITY_NEW_MEASURED = 20   # New models get low priority until user bumps them up
UNKNOWN_MODEL_PRIORITY        = 5   # Models not in config at all get very low priority

# Fallback context lengths to try when native ctx is too large for VRAM.
# Ordered from largest to smallest — first one that fits wins.
CTX_FALLBACKS = [131072, 65536, 32768, 16384, 8192, 4096, 2048]

# Default context length for models without a native context_length defined.
DEFAULT_CTX = 131072  # 128k tokens — reasonable starting point


# ---------------------------------------------------------------------------
# GPU / VRAM Helpers
# ---------------------------------------------------------------------------

def get_gpu_baseline():
    """Get VRAM used by non-Ollama processes. Returns bytes."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None

        total_baseline = 0.0
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                name = parts[1].lower()
                used_mb = float(parts[2]) * 1024 * 1024  # MiB -> bytes
                if "ollama" not in name and "nvprocess" not in name:
                    total_baseline += used_mb
            except (ValueError, IndexError):
                continue

        return int(total_baseline)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def format_bytes(b):
    """Format bytes as human-readable string."""
    if b is None or b == 0:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(b) < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"


def get_actual_gpu_total():
    """Get actual GPU total VRAM from nvidia-smi, fall back to env override."""
    global GPU_TOTAL_VRAM_BYTES
    if GPU_TOTAL_VRAM_BYTES:
        return GPU_TOTAL_VRAM_BYTES

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            total_mb = float(lines[0])
            GPU_TOTAL_VRAM_BYTES = int(total_mb * 1024 * 1024)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not GPU_TOTAL_VRAM_BYTES:
        print("[warn] Could not detect GPU total VRAM. Using default 32 GB.")
        GPU_TOTAL_VRAM_BYTES = 32 * (1024 ** 3)

    return GPU_TOTAL_VRAM_BYTES


def get_real_available_vram(loaded_ollama_bytes=None):
    """Get real available VRAM for Ollama models."""
    gpu_total = get_actual_gpu_total()
    baseline = get_gpu_baseline() or 0
    ollama_used = loaded_ollama_bytes or 0
    return gpu_total - baseline - ollama_used


def print_gpu_status(loaded_ollama_bytes=None):
    """Print GPU status summary."""
    gpu_total = get_actual_gpu_total()
    baseline = get_gpu_baseline()

    if baseline is None:
        print("[warn] Could not detect non-Ollama VRAM usage. Assuming 0.")
        baseline = 0

    ollama_used = loaded_ollama_bytes or 0
    real_available = gpu_total - baseline - ollama_used

    print(f"GPU Total:       {format_bytes(gpu_total)}")
    if baseline > 0:
        print(f"Other Processes: {format_bytes(baseline)}")
    print(f"Ollama Models:   {format_bytes(ollama_used)}")
    print(f"Real Available:  {format_bytes(real_available)}")


# ---------------------------------------------------------------------------
# Ollama API Helpers
# ---------------------------------------------------------------------------

def _make_request(url, data=None):
    """Make a request to Ollama backend with optional auth."""
    req = urlrequest.Request(url, data=data)
    if OLLAMA_API_KEY:
        req.add_header("Authorization", f"Bearer {OLLAMA_API_KEY}")

    try:
        resp = urlrequest.urlopen(req, timeout=120)
        return json.loads(resp.read())
    except (URLError, OSError, ValueError) as e:
        print(f"[error] Request to {url} failed: {e}", file=sys.stderr)
        return None


def get_local_models():
    """List all models currently available in Ollama (local + pulled)."""
    data = _make_request(f"{TARGET_HOST}/api/tags")
    if not data or "models" not in data:
        return []
    return [m["name"] for m in data.get("models", [])]


# Models known to be embedding-only and don't support /api/generate.
EMBEDDING_ONLY_MODELS = {
    "nomic-embed-text",
    "nomic-b1",
    "snowflake-arctic-embed",
    "mxbai-embed-large",
}


def is_embedding_only(model_name):
    """Check if a model name looks like it's an embedding-only model."""
    base = model_name.split(":")[0].lower().replace(" ", "-").replace("_", "-")
    for embed in EMBEDDING_ONLY_MODELS:
        if embed in base:
            return True
    return False


# ---------------------------------------------------------------------------
# Model Context Length Discovery (via /api/show)
# ---------------------------------------------------------------------------

def get_model_context_length(model_name):
    """Get the native context length of a model from Ollama's /api/show.

    Queries /api/show and extracts model_info.general.context_length, which is
    the designed maximum context for that specific model (not an arbitrary value).

    Returns:
        int — the context length in tokens, or None on failure.
    """
    show_data = _make_request(
        f"{TARGET_HOST}/api/show",
        data=json.dumps({"model": model_name}).encode()
    )
    if not show_data:
        return None

    # Try model_info.general.context_length first (standard field)
    model_info = show_data.get("model_info", {})
    general = model_info.get("general", {})
    ctx_len_str = str(general.get("context_length", ""))

    if not ctx_len_str or ctx_len_str == "0":
        # Fallback: check modelfile parameters for num_ctx
        params_text = show_data.get("parameters", "")
        if params_text and "num_ctx" in params_text:
            for line in params_text.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "num_ctx":
                    try:
                        ctx_len_str = parts[1].strip('"').strip("'")
                        break
                    except (ValueError, IndexError):
                        continue

    try:
        return int(ctx_len_str)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Unified Config Helpers
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load unified config file. Returns {"models": [...]} dict or empty list."""
    try:
        with open(config_path) as f:
            data = json.load(f)
        if "models" not in data:
            print(f"[warn] Config '{config_path}' missing 'models' key — treating as empty.")
            return {"models": []}
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"models": []}


def save_config(config_data, config_path):
    """Save unified config to file, sorted by priority DESC then vram_bytes DESC."""
    models = config_data.get("models", [])

    # Sort: highest priority first; ties broken by larger VRAM first
    models.sort(key=lambda m: (m["priority"], m["vram_bytes"]), reverse=True)
    config_data["models"] = models

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(models)} model(s) to '{config_path}' (sorted by priority):")
    for m in models:
        name = m["name"]
        pri = m["priority"]
        vram = format_bytes(m.get("vram_bytes", 0))
        ctx_info = f" @ {m.get('ctx_len', '?')} tokens" if "ctx_len" in m else ""
        print(f"  [{pri:3d}] {name}: {vram}{ctx_info}")

    print(f"\nTo use with the proxy, set:")
    print(f"  export MODEL_CONFIG_FILE='{config_path}'")


def model_name_to_key(name):
    """Convert a full Ollama model name to a config key (lowercase, no tag)."""
    base = name.split(":")[0] if ":" in name else name
    return base.lower().replace(" ", "-").replace("_", "-")


# ---------------------------------------------------------------------------
# Core Measurement
# ---------------------------------------------------------------------------

def measure_model(model_name):
    """Measure VRAM usage for a model at its native context length.

    Uses /api/show to get the model's designed context_length, then measures
    VRAM at that ctx. If it fails (too much VRAM), falls back to successively
    smaller contexts until one works.

    Args:
        model_name: Full model name with tag (e.g., 'TakacsAI-Coder-256k:latest')

    Returns:
        Tuple of (vram_bytes, ctx_len) on success, or (None, None) on failure.
    """
    # Step 1: Get the model's native context length from /api/show
    native_ctx = get_model_context_length(model_name)
    if native_ctx and native_ctx > 0:
        print(f"  Native context_length: {native_ctx:,} tokens")

    # Step 2: Try measuring at native ctx, then fall back to smaller contexts
    ctxs_to_try = []
    if native_ctx and native_ctx > 0:
        ctxs_to_try.append(native_ctx)
    # Always include the standard fallback chain (in case native is too large or unknown)
    for fb in CTX_FALLBACKS:
        if fb not in ctxs_to_try:
            ctxs_to_try.append(fb)

    last_result = None
    for try_ctx in ctxs_to_try:
        print(f"\n{'='*60}")
        print(f"Measuring: {model_name} @ ctx={try_ctx:,} (attempt)")
        print(f"{'='*60}")

        # Step 1: Trigger model load via generate API with keep_alive=-1
        # keep_alive=-1 prevents Ollama from auto-evicting the loaded model
        # when subsequent models are measured in a batch run.
        print(f"[1/3] Loading '{model_name}' with num_ctx={try_ctx:,}...")
        data = _make_request(
            f"{TARGET_HOST}/api/generate",
            data=json.dumps({
                "model": model_name,
                "prompt": "test",
                "stream": False,
                "options": {"num_ctx": try_ctx},
                "keep_alive": -1
            }).encode()
        )

        if not data:
            print(f"  ✗ Failed to load at ctx={try_ctx:,} (API error). Trying next...")
            last_result = None
            continue

        # Step 2: Wait for VRAM to settle
        print(f"[2/3] Waiting 5s for VRAM allocation...")
        time.sleep(5)

        # Step 3: Read actual total VRAM from /api/ps
        print(f"[3/3] Reading VRAM usage from /api/ps...")
        ps_data = _make_request(f"{TARGET_HOST}/api/ps")
        if not ps_data or "models" not in ps_data:
            print(f"  ✗ Failed to read VRAM from /api/ps. Trying next ctx...")
            last_result = None
            continue

        # Ollama's /api/ps reports size_vram (total VRAM including KV cache) but NOT ctx_len.
        # The ctx we used IS what was actually configured for this measurement.
        matched = False
        for m in ps_data["models"]:
            if model_name in m.get("name", "") or m.get("digest", "") == data.get("model_digest", ""):
                size = int(m.get("size_vram", 0))
                print(f"  ✓ Total VRAM (incl. KV cache): {size / 1e9:.2f} GB ({size:,} bytes)")
                print(f"  ✓ Context length used: {try_ctx:,}")
                last_result = (size, try_ctx)

                # If this was the native ctx, we're done — no need to fall back further.
                if try_ctx == native_ctx:
                    return size, try_ctx
                matched = True
                break

        if not matched and last_result is None:
            print(f"  ✗ Model evicted before measurement at ctx={try_ctx:,}. Trying next...")
        elif last_result is not None:
            # Successfully measured at a fallback ctx — report it but don't try further.
            print(f"  ✓ Measured at fallback ctx={try_ctx:,} (native was {native_ctx or '?'})")
            return size, try_ctx

    # All attempts exhausted
    if last_result is None:
        print(f"\n✗ Failed to measure '{model_name}' — all context lengths too large for VRAM.")
    return last_result


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def cleanup_mode(config_path):
    """Remove entries for models that no longer exist locally."""
    unified = load_config(config_path)
    available = get_local_models()

    if not available:
        print("[warn] Could not list Ollama models. Aborting cleanup.")
        return

    # Build set of keys from local models (strip tags for comparison)
    available_keys = {model_name_to_key(m) for m in available}

    removed = []
    kept = []

    for entry in unified["models"]:
        key = model_name_to_key(entry["name"])
        if key in available_keys:
            kept.append(entry)
        else:
            # Try fuzzy match by base name
            base = key.split("-")[0] if "-" in key else key
            found = any(base in ak for ak in available_keys)
            if not found:
                removed.append(entry)

    if not removed:
        print("No stale entries to remove. All measured models are still available.")
        return

    print(f"\nRemoved {len(removed)} stale entry/entries:")
    for m in removed:
        print(f"  ✗ {m['name']} ({format_bytes(m.get('vram_bytes', 0))})")

    unified["models"] = kept
    save_config(unified, config_path)


def benchmark_all_mode(config_path, force_recalc=False):
    """Benchmark every local Ollama model at its native context length.

    By default only measures models NOT already in the config file.
    Use --force to recalculate all entries even if they exist.
    """
    available = get_local_models()
    if not available:
        print("[error] No models found in Ollama.")
        return

    unified = load_config(config_path)

    # Build set of keys already in config (from base name, strip tag)
    measured_keys = {model_name_to_key(m["name"]) for m in unified["models"]}

    if force_recalc:
        to_measure = available
        print("Force recalc mode — measuring ALL models at their native context lengths.")
    else:
        # Only measure models not already in the config
        new_models = [m for m in available if model_name_to_key(m) not in measured_keys]
        skip_count = len(available) - len(new_models)
        print(f"Found {len(available)} local models.")
        print(f"  Measuring: {len(new_models)} (not yet in config)")
        print(f"  Skipping:  {skip_count} (already in config)")
        to_measure = new_models

    for model_name in to_measure:
        if is_embedding_only(model_name):
            key = model_name_to_key(model_name)
            print(f"\nSkipping embedding-only model: {model_name} (key: '{key}')")
            continue

        key = model_name_to_key(model_name)
        native_ctx = get_model_context_length(model_name)
        native_info = f" (native ctx={native_ctx:,})" if native_ctx and native_ctx > 0 else " (unknown native ctx)"
        print(f"\n{'#'*60}")
        print(f"# Measuring: {model_name}{native_info} (key: '{key}')")
        print(f"{'#'*60}")

        result = measure_model(model_name)
        if result and len(result) == 2 and result[0] is not None:
            size, ctx_len = result
            # Update or add in unified config
            found = False
            for m in unified["models"]:
                if model_name_to_key(m["name"]) == key:
                    m["vram_bytes"] = size
                    m["ctx_len"] = ctx_len
                    print(f"  ✓ Updated '{key}' → {size / 1e9:.2f} GB @ {ctx_len:,} tokens")
                    found = True
                    break
            if not found:
                unified["models"].append({
                    "name": key,
                    "vram_bytes": size,
                    "ctx_len": ctx_len,
                    "priority": DEFAULT_PRIORITY_NEW_MEASURED
                })
                print(f"  ✓ Added '{key}' → {size / 1e9:.2f} GB @ {ctx_len:,} tokens (pri={DEFAULT_PRIORITY_NEW_MEASURED})")
        else:
            print(f"  ✗ Failed — skipping.")

    save_config(unified, config_path)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure VRAM usage for Ollama models at their native context lengths."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a", "--all", dest="benchmark_all", action="store_true",
                       help="Benchmark ALL local Ollama models at their native context lengths")
    group.add_argument("--cleanup", action="store_true",
                       help="Remove stale entries from config file for models no longer available")

    parser.add_argument("-o", "--output", default="ollama_model_registry.json",
                        help="Config file path (default: ollama_model_registry.json)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force recalculation of ALL models (use with --all)")

    args = parser.parse_args()

    config_path = args.output or "ollama_model_registry.json"

    if args.benchmark_all:
        benchmark_all_mode(config_path, force_recalc=args.force)
    elif args.cleanup:
        cleanup_mode(config_path)


if __name__ == "__main__":
    main()