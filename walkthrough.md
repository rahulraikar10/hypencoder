import os
import json
import re
import time
import asyncio
import aiohttp
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ─── Gauss LLM Client ───────────────────────────────────────────────────────────

class GaussLLMClient:
    """Async client for Gauss LLM API."""

    def __init__(
        self,
        client_key: Optional[str] = None,
        pass_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        user_email: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.94,
        repetition_penalty: float = 1.04,
        request_timeout: int = 60,
    ):
        self.client_key = client_key or os.getenv("GAUSS_CLIENT_KEY", "")
        self.pass_key = pass_key or os.getenv("GAUSS_PASS_KEY", "")
        self.endpoint_url = endpoint_url or os.getenv("GAUSS_ENDPOINT_URL", "")
        self.user_email = user_email or os.getenv("GAUSS_USER_EMAIL", "")

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.request_timeout = request_timeout

        self._headers = {
            "x-generative-ai-client": self.client_key,
            "x-openapi-token": self.pass_key,
            "x-generative-ai-user-email": self.user_email,
            "Content-Type": "application/json"
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._model_id: Optional[str] = None  # cached after first successful fetch

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_model_id(self) -> Optional[str]:
        """Fetch available model ID from Gauss API (cached after first success)."""
        if self._model_id:
            return self._model_id
        if not self.endpoint_url:
            return None
        models_endpoint = f"{self.endpoint_url}/openapi/chat/v1/models"
        try:
            session = await self._get_session()
            async with session.get(models_endpoint, headers=self._headers) as resp:
                if resp.status == 200:
                    models = await resp.json()
                    if models and len(models) > 0:
                        self._model_id = models[0].get("modelId")
                        return self._model_id
                else:
                    body = await resp.text()
                    print(f"Error fetching model ID: HTTP {resp.status}: {body[:200]}")
        except asyncio.TimeoutError:
            print(f"Error fetching model ID: request timed out after {self.request_timeout}s")
        except Exception as e:
            print(f"Error fetching model ID: {e}")
        return None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response from Gauss LLM."""
        if not self.endpoint_url or not self.client_key or not self.pass_key:
            return "Error: GAUSS credentials not configured. Check .env file."

        model_id = await self._fetch_model_id()
        if not model_id:
            return "Error: Could not fetch model ID from Gauss API."

        body = {
            "modelIds": [model_id],
            "contents": [prompt],
            "isStream": False,
            "disableReasoning": True,
            "llmConfig": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "enable_thinking": False
            }
        }

        if system_prompt:
            body["systemPrompt"] = system_prompt

        endpoint = f"{self.endpoint_url}/openapi/chat/v1/messages"

        try:
            session = await self._get_session()
            async with session.post(endpoint, headers=self._headers, json=body) as resp:
                raw_text = await resp.text()
                print(f"    [GAUSS DEBUG] Raw response ({resp.status}): {raw_text[:500]}...")

                if resp.status == 200:
                    data = await resp.json()
                    print(f"    [GAUSS DEBUG] Parsed data: {data}")

                    if data is None:
                        return "Error: Empty response from Gauss API."

                    status = data.get("status")
                    response_code = data.get("responseCode")
                    content = data.get("content")

                    print(f"    [GAUSS DEBUG] status={status}, responseCode={response_code}, content_len={len(content) if content else 0}")

                    if not content:
                        content = data.get("answer") or data.get("response") or data.get("text") or data.get("message")

                    if status == "SUCCESS" and response_code == "R20000":
                        return content or "No answer generated"
                    else:
                        return f"API Error: status={status}, code={response_code}"
                else:
                    return f"Error: Gauss API error ({resp.status}): {raw_text[:200]}"
        except asyncio.TimeoutError:
            return f"Error: request timed out after {self.request_timeout}s"
        except Exception as e:
            print(f"    [GAUSS DEBUG] Exception: {e}")
            return f"Error calling Gauss API: {e}"


# Deterministic settings: the prompt requires verbatim fidelity to source code,
# so we use temperature=0.0 (no creative drift, no fabricated code).
gauss_client = GaussLLMClient(
    temperature=0.0,
    top_p=0.94,
    repetition_penalty=1.04,
    request_timeout=60,
)

# ─── Script Configuration ───────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CAPSULE_PATH = SCRIPT_DIR.parent / 'capsule1'
OUTPUT_PATH = SCRIPT_DIR.parent / 'output' / 'capsules-chunks.md'

EXCLUDED_FOLDERS = ['app-context', '__mocked_data__', '__test__', 'assets', 'svg']

BATCH_SIZE = 1             # Files per request
RATE_LIMIT = 1             # Requests per second
RATE_WINDOW = 1            # Seconds for rate limit window
MAX_RETRIES = 3            # Retry count for failed requests
BASE_RETRY_DELAY = 5       # Base delay between retries (seconds)

# Rate limiting state
_request_timestamps = []

# ─── Prompt ───────────────────────────────────────────────────────────────────

CHUNKING_PROMPT = """You are extracting code chunks from a TypeScript file for a search index.

IMPORTANT: 
- Process the file and extract chunks for code functions/methods
- Each chunk must reference the correct source file path
- Never invent, rename, or guess code that isn't present in the file

=== HARD RULES (breaking any of these is a failure) ===
1. NEVER output a chunk for import statements. Skip them entirely, always.
2. NEVER output a chunk for a bare `export default X;` line with no body.
3. Group ALL types/interfaces into exactly one chunk named `types_and_interfaces`. If none exist, skip it.
4. Only chunk a function/method with 3+ lines of real logic. Skip one-liners, empty functions, trivial getters/setters.
5. Every line you output must exist in the FILE CONTENT. Do not fabricate function names, params, or logic.
6. If the file has no qualifying functions, output exactly: NO_CHUNKS

=== CONDENSING (only for chunks 15+ lines; leave shorter chunks untouched) ===
Fold control flow into one line each:
  if/else   -> // if CONDITION -> OUTCOME | else -> OUTCOME
  switch    -> // switch(VAR): CASE->RESULT | CASE->RESULT
  try/catch -> // try: ACTION | catch: ERROR HANDLING
Collapse null/guard checks and repeated param destructuring into one comment line.
Drop dead code and redundant intermediate variables. Keep: real business logic, key calls, return values, error codes/constants. Target 10-20 lines.

=== INNER FUNCTIONS ===
- under 15 lines -> fold: // inner fn NAME(PARAMS) -> PURPOSE
- 15+ lines -> own chunk; in the outer chunk write: // calls NAME() -> see chunk: CHUNK_NAME

=== OUTPUT FORMAT (exact, nothing else) ===

## CHUNK: SNAKE_CASE_NAME
**File:** FILE_PATH
**Summary:** 2-3 dense sentences — what it does/purpose, key inputs/outputs, notable error codes or edge cases in English

```ts
CONDENSED_CODE
```
---

=== INPUT FILE ===

FILE PATH: {file_path}
FILE CONTENT:
{file_content}

=== END OF INPUT ===

Now extract chunks from the file following the rules above.
"""

# ─── File discovery ───────────────────────────────────────────────────────────

def find_ts_files(dir_path, file_list=None):
    """Find all .ts files recursively, excluding specified folders."""
    if file_list is None:
        file_list = []
    try:
        files = os.listdir(dir_path)
    except PermissionError:
        return file_list
    for file in files:
        file_path = os.path.join(dir_path, file)
        if os.path.isdir(file_path):
            if file in EXCLUDED_FOLDERS:
                continue
            file_list = find_ts_files(file_path, file_list)
        elif file.endswith('.ts') and not file.endswith('.d.ts'):
            file_list.append(file_path)
    return file_list

# ─── Rate Limiting ──────────────────────────────────────────────────────────────

async def enforce_rate_limit():
    """Ensure we don't exceed RATE_LIMIT requests per RATE_WINDOW seconds."""
    global _request_timestamps

    now = time.time()
    _request_timestamps = [ts for ts in _request_timestamps if now - ts < RATE_WINDOW]

    if len(_request_timestamps) >= RATE_LIMIT:
        oldest = _request_timestamps[0]
        wait_time = RATE_WINDOW - (now - oldest) + 1
        if wait_time > 0:
            tqdm.write(f'    [RATE LIMIT] Waiting {wait_time:.1f}s before next request...')
            await asyncio.sleep(wait_time)
            now = time.time()
            _request_timestamps = [ts for ts in _request_timestamps if now - ts < RATE_WINDOW]

    _request_timestamps.append(time.time())

# ─── LLM call ─────────────────────────────────────────────────────────────────

async def call_llm(prompt: str, file_label: str = '') -> str:
    """Call Gauss LLM with rate limiting and retry logic."""
    await enforce_rate_limit()

    t0 = time.time()
    tqdm.write(f'    [LLM] Request{" for " + file_label if file_label else ""}')

    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            response = await gauss_client.generate(prompt=prompt)
            elapsed = time.time() - t0

            # Retry on transient/server errors
            if '500' in response or '502' in response or 'Bad Gateway' in response or 'timeout' in response.lower():
                last_error = Exception(response)
                retry_delay = BASE_RETRY_DELAY * (attempt + 1)
                tqdm.write(f'    [LLM] ✗ Server error, retrying in {retry_delay}s... (attempt {attempt + 1}/{MAX_RETRIES})')
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(retry_delay)
                continue

            # Non-retryable error (bad creds, bad request, etc.) - fail fast
            if response.startswith("Error:") or response.startswith("API Error:"):
                tqdm.write(f'    [LLM] ✗ {response}')
                return response

            tqdm.write(f'    [LLM] ✓ done in {elapsed:.1f}s | response_len={len(response)}')
            return response

        except Exception as e:
            last_error = e
            retry_delay = BASE_RETRY_DELAY * (attempt + 1)
            tqdm.write(f'    [LLM] ✗ Error (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {retry_delay}s...')
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(retry_delay)

    tqdm.write(f'    [LLM] ✗ All {MAX_RETRIES} retries failed')
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM call failed with no captured error")

# ─── Response parser ───────────────────────────────────────────────────────────

def parse_chunks_from_response(response: str) -> list[dict]:
    """
    Parse LLM response into a list of chunk dicts.
    Handles multi-file responses by extracting file path from each chunk.
    """
    chunks = []

    raw_blocks = re.split(r'(?=## CHUNK:)', response)

    for block in raw_blocks:
        block = block.strip()
        if not block.startswith('## CHUNK:'):
            continue

        name_match = re.search(r'^## CHUNK:\s*(\S+)', block, re.MULTILINE)
        if not name_match:
            tqdm.write(f'    [PARSE] ⚠ block has no chunk name, skipping')
            continue
        chunk_name = name_match.group(1)

        file_match = re.search(r'\*\*File:\*\*\s*(.+?)(?:\n|$)', block)
        summary_match = re.search(r'\*\*Summary:\*\*\s*(.+?)(?:\n\s*```|\n\n|$)', block, re.DOTALL)
        code_match = re.search(r'```(?:ts|typescript)?\s*\n(.+?)\n```', block, re.DOTALL)

        if not file_match:
            tqdm.write(f'    [PARSE] ⚠ chunk {chunk_name!r}: no **File:** line, skipping')
            continue
        if not summary_match:
            tqdm.write(f'    [PARSE] ⚠ chunk {chunk_name!r}: no **Summary:** line, skipping')
            continue
        if not code_match:
            tqdm.write(f'    [PARSE] ⚠ chunk {chunk_name!r}: no ```ts``` block, skipping')
            continue

        chunks.append({
            'chunk_name': chunk_name,
            'file': file_match.group(1).strip(),
            'summary': summary_match.group(1).strip(),
            'code': code_match.group(1).strip(),
        })

    return chunks

# ─── Batch processing ──────────────────────────────────────────────────────────

async def process_batch(file_paths: list[str]) -> tuple[list[dict], list[str]]:
    """
    Process a single file (1 file per batch).
    Returns (all_chunks, errors) where all_chunks is list of chunk dicts
    and errors is list of error messages.
    """
    errors = []

    file_path = file_paths[0]

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        try:
            relative_path = str(Path(file_path).relative_to(CAPSULE_PATH.parent))
        except ValueError:
            relative_path = file_path

        prompt = CHUNKING_PROMPT.format(
            file_path=relative_path,
            file_content=content,
        )

        tqdm.write(f'    [BATCH] Processing file: {relative_path}')
        tqdm.write(f'    [BATCH] Total prompt length: {len(prompt)} chars')

        response = await call_llm(prompt, file_label=relative_path)

        if not response or response.startswith("Error:") or response.startswith("API Error:"):
            errors.append(f"LLM error for file [{relative_path}]: {response}")
            return [], errors

        if response.strip() == 'NO_CHUNKS':
            tqdm.write(f'    [BATCH] → No qualifying chunks in file')
            return [], errors

        parsed_chunks = parse_chunks_from_response(response)
        tqdm.write(f'    [BATCH] → parsed {len(parsed_chunks)} chunk(s)')
        return parsed_chunks, errors

    except Exception as e:
        msg = f'## ERROR processing {file_path}: {e}\n\n'
        errors.append(msg)
        tqdm.write(f'    [BATCH] ✗ {e}')
        return [], errors

# ─── Output helpers ────────────────────────────────────────────────────────────

def generate_chunk_id(file_path: str, chunk_name: str) -> str:
    """Stable URI-style chunk identifier used in JSONL and as HDF5 attribute."""
    return f"ts://{file_path}#{chunk_name}"


def _chunk_id_to_hdf5_key(file_path: str, chunk_name: str) -> str:
    """
    Convert a chunk's file path + name into a valid HDF5 dataset path.
    """
    safe_file = file_path.replace('\\', '/').lstrip('/')
    safe_name = chunk_name.replace('/', '_')
    return f"{safe_file}/{safe_name}"


def append_jsonl_output(chunks: list[dict], output_path: Path):
    with open(output_path, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            entry = {
                'id': generate_chunk_id(chunk['file'], chunk['chunk_name']),
                'text': chunk['summary'],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def write_hdf5_output(all_chunks: list[dict], hdf5_path: Path):
    """Write all chunks to HDF5."""
    str_dtype = h5py.string_dtype()
    with h5py.File(hdf5_path, 'w') as h5f:
        for chunk in all_chunks:
            hdf5_key = _chunk_id_to_hdf5_key(chunk['file'], chunk['chunk_name'])
            chunk_id = generate_chunk_id(chunk['file'], chunk['chunk_name'])
            ds = h5f.create_dataset(
                hdf5_key,
                data=chunk['code'].encode('utf-8'),
                dtype=str_dtype,
            )
            ds.attrs['chunk_id'] = chunk_id


def post_process_output(output_path: Path) -> list[str]:
    """Tidy up the raw accumulated markdown and ensure every chunk has --- separators."""
    print('Post-processing output file...')

    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()

    first_chunk_idx = content.find('## CHUNK:')
    if first_chunk_idx == -1:
        print('  No chunks found, skipping post-processing')
        return []
    content = content[first_chunk_idx:]

    content = re.sub(r'```\s*\n---', '```\n---', content)
    content = re.sub(r'\n---\n\s*\n---\n', '\n---\n', content)

    raw_blocks = re.split(r'(?=## CHUNK:)', content)
    processed_chunks = []
    for block in raw_blocks:
        block = block.strip()
        if not block.startswith('## CHUNK:'):
            continue
        if not block.endswith('---'):
            block = block.rstrip() + '\n---'
        processed_chunks.append(block)

    final_content = '\n\n'.join(processed_chunks)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f'  Post-processing complete. {len(processed_chunks)} chunks.')
    return processed_chunks

# ─── Startup connectivity check ────────────────────────────────────────────────

async def check_connection() -> bool:
    """Verify Gauss credentials/endpoint and connectivity before starting the batch loop."""
    print('Checking Gauss connection...')
    print(f'  endpoint:         {gauss_client.endpoint_url or "(not set)"}')
    print(f'  client_key set:   {bool(gauss_client.client_key)}')
    print(f'  pass_key set:     {bool(gauss_client.pass_key)}')
    print(f'  user_email set:   {bool(gauss_client.user_email)}')

    if not gauss_client.endpoint_url or not gauss_client.client_key or not gauss_client.pass_key:
        print('  ✗ Missing required credentials. Check your .env file '
              '(GAUSS_CLIENT_KEY, GAUSS_PASS_KEY, GAUSS_ENDPOINT_URL).')
        return False

    try:
        model_id = await gauss_client._fetch_model_id()
    except Exception as e:
        print(f'  ✗ Connection check raised an exception: {e}')
        return False

    if not model_id:
        print('  ✗ Could not fetch a model ID. Endpoint may be unreachable, '
              'credentials may be invalid, or the request timed out.')
        return False

    print(f'  ✓ Connected. model_id={model_id}')
    return True

# ─── Main (async) ──────────────────────────────────────────────────────────────

async def main_async():
    start_time = time.time()

    print(f'Starting chunking for all capsules (GAUSS LLM, {BATCH_SIZE}-file batches)')
    print(f'Capsule path:     {CAPSULE_PATH}')
    print(f'Output path:      {OUTPUT_PATH}')
    print(f'Excluded folders: {", ".join(EXCLUDED_FOLDERS)}')
    print(f'Batch size:      