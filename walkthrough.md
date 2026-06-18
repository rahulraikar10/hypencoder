"""
Quick diagnostic for llama.cpp server at http://107.109.107.68:8080
Run this before the chunker to find out exactly where it's hanging.
"""

import sys
import time
import socket
import requests
import openai

BASE_URL = "http://107.109.107.68:8080"
MODEL = "qwen:3.5:9B"
TIMEOUT = 30  # seconds for each test

print("=" * 60)
print("llama.cpp server diagnostics")
print(f"Target: {BASE_URL}")
print("=" * 60)

# ── Test 1: Raw TCP connectivity ──────────────────────────────
print("\n[1] TCP connectivity...")
try:
    sock = socket.create_connection(("107.109.107.68", 8080), timeout=5)
    sock.close()
    print("    ✓ Port 8080 is reachable")
except Exception as e:
    print(f"    ✗ Cannot reach port 8080: {e}")
    print("    → Server is down or wrong IP/port. Nothing else will work.")
    sys.exit(1)

# ── Test 2: GET /health ────────────────────────────────────────
print("\n[2] GET /health ...")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    print(f"    status: {r.status_code}")
    print(f"    body:   {r.text[:300]}")
except Exception as e:
    print(f"    ✗ {e}")

# ── Test 3: GET /v1/models ─────────────────────────────────────
print("\n[3] GET /v1/models ...")
try:
    r = requests.get(f"{BASE_URL}/v1/models", timeout=TIMEOUT)
    print(f"    status: {r.status_code}")
    print(f"    body:   {r.text[:500]}")
except Exception as e:
    print(f"    ✗ {e}")

# ── Test 4: Raw POST to /v1/chat/completions (no SDK) ─────────
print("\n[4] Raw POST /v1/chat/completions (requests, 30s timeout) ...")
t0 = time.time()
try:
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say: OK"}],
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=TIMEOUT,
    )
    elapsed = time.time() - t0
    print(f"    status:  {r.status_code}  ({elapsed:.1f}s)")
    print(f"    body:    {r.text[:500]}")
except requests.exceptions.Timeout:
    elapsed = time.time() - t0
    print(f"    ✗ Timed out after {elapsed:.1f}s")
    print("    → Server is up but not responding to chat completions.")
    print("      Possible causes: model still loading, slots all busy, wrong endpoint.")
except Exception as e:
    elapsed = time.time() - t0
    print(f"    ✗ {e} ({elapsed:.1f}s)")

# ── Test 5: Raw POST to /v1/completions (legacy) ──────────────
print("\n[5] Raw POST /v1/completions (legacy, 30s timeout) ...")
t0 = time.time()
try:
    r = requests.post(
        f"{BASE_URL}/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "prompt": "Say: OK",
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=TIMEOUT,
    )
    elapsed = time.time() - t0
    print(f"    status:  {r.status_code}  ({elapsed:.1f}s)")
    print(f"    body:    {r.text[:500]}")
except requests.exceptions.Timeout:
    elapsed = time.time() - t0
    print(f"    ✗ Timed out after {elapsed:.1f}s")
except Exception as e:
    elapsed = time.time() - t0
    print(f"    ✗ {e} ({elapsed:.1f}s)")

# ── Test 6: OpenAI SDK chat completions ───────────────────────
print("\n[6] OpenAI SDK — chat completions (30s timeout) ...")
t0 = time.time()
try:
    client = openai.OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="sk-no-key-required",
        timeout=30.0,
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say: OK"}],
        max_tokens=10,
        temperature=0,
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content
    finish = resp.choices[0].finish_reason
    print(f"    ✓ response: {text!r}  finish={finish}  ({elapsed:.1f}s)")
except openai.APITimeoutError:
    elapsed = time.time() - t0
    print(f"    ✗ SDK timeout after {elapsed:.1f}s")
except openai.APIConnectionError as e:
    elapsed = time.time() - t0
    print(f"    ✗ SDK connection error: {e} ({elapsed:.1f}s)")
except openai.APIStatusError as e:
    elapsed = time.time() - t0
    print(f"    ✗ SDK API error {e.status_code}: {e.message} ({elapsed:.1f}s)")
except Exception as e:
    elapsed = time.time() - t0
    print(f"    ✗ {e} ({elapsed:.1f}s)")

print("\n" + "=" * 60)
print("Done. Paste the output above to diagnose the issue.")
print("=" * 60)
