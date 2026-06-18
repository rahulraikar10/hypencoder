import os
import json
import re
import time
import openai
import h5py
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
# FIX: model string updated to match the actual model ID reported by /v1/models.
# llama.cpp ignores this field and uses whatever is loaded, but keeping it
# accurate avoids confusion when reading logs.
LLM_MODEL = 'Qwen3.5-9B-Q8_0.gguf'

client = openai.OpenAI(
    base_url="http://107.109.107.68:8080/v1",
    api_key="sk-no-key-required",
    timeout=600.0,
)

CAPSULE_NAME = 'bigOven'
SCRIPT_DIR = Path(__file__).parent
CAPSULE_PATH = SCRIPT_DIR.parent / 'capsules' / CAPSULE_NAME
OUTPUT_PATH = SCRIPT_DIR.parent / 'output' / f'{CAPSULE_NAME.lower()}-chunks.md'

EXCLUDED_FOLDERS = ['app-context', '__mocked_data__', '__test__']

MAX_RETRIES = 3
BASE_RETRY_DELAY = 10      # seconds
MAX_FILE_CHARS = 12_000    # truncate huge files to avoid context overflow

# ─── Prompt ───────────────────────────────────────────────────────────────────
# NOTE: uses {file_path} and {file_content} as .format() placeholders.
# All literal braces in the prompt text are escaped as {{ / }}.
CHUNKING_PROMPT = """You are extracting code chunks from a TypeScript file for a search index. Copy code verbatim from FILE CONTENT below — never invent, rename, or guess code that isn't present in the file.

FILE PATH: {file_path}

FILE CONTENT:
{file_content}

=== HARD RULES (breaking any of these is a failure) ===
1. NEVER output a chunk for import statements. Not even an "imports" chunk. Skip them entirely, always.
2. NEVER output a chunk for a bare `export default X;` line with no body.
3. Group ALL types/interfaces into exactly one chunk named `types_and_interfaces`. If none exist, skip it.
4. Only chunk a function/method with 3+ lines of real logic. Skip one-liners, empty functions, trivial getters/setters.
5. Every line you output must exist in FILE CONTENT. Do not fabricate function names, params, or logic.
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
**Summary:** 2-3 dense sentences — what it does/purpose, key inputs/outputs, notable error codes or edge cases.

```ts
CONDENSED_CODE
```
---

=== WORKED EXAMPLE ===
INPUT:
import {{ z }} from "zod";
export const MAX = 5;
function add(a: number, b: number) {{ return a + b; }}
export function validate(input: unknown) {{
  if (!input) {{ return {{ ok: false, error: "EMPTY_INPUT" }}; }}
  const parsed = z.string().safeParse(input);
  if (!parsed.success) {{ return {{ ok: false, error: "INVALID_TYPE" }}; }}
  return {{ ok: true, value: parsed.data }};
}}

CORRECT OUTPUT:
## CHUNK: validate
**File:** example.ts
**Summary:** Validates that unknown input is a non-empty string via zod's safeParse. Takes an unknown input, returns {{ok, value}} on success or {{ok:false, error}} on failure, with EMPTY_INPUT for falsy input and INVALID_TYPE for failed parsing.

```ts
export function validate(input: unknown) {{
  // guard: empty input -> return {{ok:false, error:"EMPTY_INPUT"}}
  const parsed = z.string().safeParse(input);
  // if !parsed.success -> return {{ok:false, error:"INVALID_TYPE"}} | else -> return {{ok:true, value:parsed.data}}
}}
```
---
(add() and MAX are skipped: add has <3 lines of logic, MAX isn't a function. The import line is never its own chunk.)
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


# ─── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(prompt: str, file_label: str = '') -> str:
    """
    Call the llama.cpp server via chat completions.

    FIX (thinking mode): Qwen3 ships with chain-of-thought thinking enabled by
    default. When active, the model writes all output to `reasoning_content` and
    leaves `content` empty — so the caller receives an empty string regardless of
    how many tokens are generated. Diagnosed via /v1/chat/completions returning
        {"content": "", "reasoning_content": "Thinking Process: ..."}
    Fix: pass `enable_thinking: false` through llama.cpp's `chat_template_kwargs`
    so the model skips the <think> block and writes directly to `content`.

    Fallback: if `content` is still empty after the request (e.g. a future model
    version or server config that ignores the kwarg), extract `reasoning_content`
    rather than silently returning ''.
    """
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        try:
            tqdm.write(f'    [LLM] attempt {attempt + 1}/{MAX_RETRIES + 1}'
                       f'{" for " + file_label if file_label else ""}')

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0,
                # FIX: disable Qwen3 thinking mode so output goes to `content`,
                # not `reasoning_content`. Passed as a raw body field via
                # extra_body; llama.cpp forwards it to the chat template renderer.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            elapsed = time.time() - t0

            msg = response.choices[0].message
            result = (msg.content or '').strip()

            # FIX: fallback — if content is still empty, pull from reasoning_content.
            # This should not happen once enable_thinking=False is respected, but
            # keeps the pipeline alive if the server ignores the kwarg.
            if not result:
                reasoning = getattr(msg, 'reasoning_content', None) or ''
                if reasoning:
                    result = reasoning.strip()
                    tqdm.write('    [LLM] ⚠ content was empty; fell back to '
                               'reasoning_content (thinking mode still active?)')

            finish = response.choices[0].finish_reason
            tok_in = getattr(response.usage, 'prompt_tokens', '?')
            tok_out = getattr(response.usage, 'completion_tokens', '?')

            tqdm.write(f'    [LLM] ✓ done in {elapsed:.1f}s | '
                       f'finish={finish} | tokens in={tok_in} out={tok_out} | '
                       f'response_len={len(result)}')

            if finish == 'length':
                tqdm.write('    [LLM] ⚠ finish_reason=length — output was truncated!')

            return result

        except openai.APITimeoutError:
            elapsed = time.time() - t0
            delay = BASE_RETRY_DELAY * (attempt + 1)
            last_error = Exception(f'Timeout after {elapsed:.1f}s (attempt {attempt + 1})')
            tqdm.write(f'    [LLM] ⏱ Timeout after {elapsed:.1f}s — retry in {delay}s...')
            time.sleep(delay)

        except openai.APIConnectionError as e:
            delay = BASE_RETRY_DELAY * (attempt + 1)
            last_error = Exception(f'Connection error: {e}')
            tqdm.write(f'    [LLM] 🔌 Connection error: {e} — retry in {delay}s...')
            time.sleep(delay)

        except openai.APIStatusError as e:
            if e.status_code == 429:
                delay = BASE_RETRY_DELAY * (attempt + 1)
                last_error = Exception(f'Rate limited (attempt {attempt + 1})')
                tqdm.write(f'    [LLM] 🚦 Rate limited — retry in {delay}s...')
                time.sleep(delay)
            else:
                last_error = Exception(f'API error {e.status_code}: {e.message}')
                tqdm.write(f'    [LLM] ✗ API error {e.status_code}: {e.message} — not retrying')
                break

        except Exception as e:
            last_error = Exception(f'Unexpected error: {e}')
            tqdm.write(f'    [LLM] ✗ Unexpected error: {e} — not retrying')
            break

    raise last_error


# ─── Response parser ──────────────────────────────────────────────────────────
def parse_chunks_from_response(response: str, file_path: str) -> list[dict]:
    """
    Parse LLM response into a list of chunk dicts.
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


# ─── Per-file processing ──────────────────────────────────────────────────────
def process_file(file_path: str) -> tuple[str, list[dict]]:
    """Read a TS file, call the LLM, return (raw_response, parsed_chunks)."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    char_count = len(content)
    if char_count > MAX_FILE_CHARS:
        tqdm.write(f'    [FILE] ⚠ truncating {char_count} chars → {MAX_FILE_CHARS} (set MAX_FILE_CHARS to override)')
        content = content[:MAX_FILE_CHARS] + '\n// ... [TRUNCATED]'

    try:
        relative_path = str(Path(file_path).relative_to(CAPSULE_PATH.parent))
    except ValueError:
        relative_path = file_path

    tqdm.write(f'    [FILE] {relative_path} ({char_count} chars)')

    prompt = CHUNKING_PROMPT.format(
        file_path=relative_path,
        file_content=content,
    )

    tqdm.write(f'    [FILE] prompt length: {len(prompt)} chars')

    try:
        response = call_llm(prompt, file_label=relative_path)

        if response.strip() == 'NO_CHUNKS':
            tqdm.write('    [FILE] → model returned NO_CHUNKS')
            return response, []

        parsed_chunks = parse_chunks_from_response(response, relative_path)
        tqdm.write(f'    [FILE] → parsed {len(parsed_chunks)} chunk(s)')
        return response, parsed_chunks

    except Exception as e:
        msg = f'## ERROR processing {relative_path}: {e}\n\n'
        tqdm.write(f'    [FILE] ✗ {e}')
        return msg, []


# ─── Output helpers ───────────────────────────────────────────────────────────
def generate_chunk_id(file_path: str, chunk_name: str) -> str:
    """Stable URI-style chunk identifier used in JSONL and as HDF5 attribute."""
    return f"ts://{file_path}#{chunk_name}"


def _chunk_id_to_hdf5_key(file_path: str, chunk_name: str) -> str:
    """
    Convert a chunk's file path + name into a valid HDF5 dataset path.

    FIX: the chunk_id (ts://path/file.ts#name) cannot be used directly as an
    HDF5 dataset name. h5py treats '/' as a path separator, so '://' produces
    an empty path component that raises ValueError. '#' is technically legal but
    confusing. Instead we build a clean hierarchical key from the parts we
    already have: 'path/to/file.ts/chunk_name'.
    The original chunk_id URI is stored as a dataset attribute for round-tripping.
    """
    # Normalise any backslashes (Windows paths) and strip leading slashes
    safe_file = file_path.replace('\\', '/').lstrip('/')
    safe_name = chunk_name.replace('/', '_')   # chunk names should never have '/' but guard anyway
    return f"{safe_file}/{safe_name}"


def append_jsonl_output(chunks: list[dict], output_path: Path):
    with open(output_path, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            entry = {
                'id': generate_chunk_id(chunk['file'], chunk['chunk_name']),
                'text': chunk['summary'],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def write_hdf5_output(all_chunks: list[dict], hdf5_path: Path, capsule_name: str):
    """
    Write all chunks to HDF5.

    Each dataset is keyed by a clean hierarchical path (file/chunk_name) and
    stores the condensed TypeScript code as a UTF-8 string. The full chunk_id
    URI is preserved as a dataset attribute so consumers can correlate records
    back to the JSONL embeddings file.
    """
    str_dtype = h5py.string_dtype()
    with h5py.File(hdf5_path, 'w') as h5f:
        capsule_group = h5f.create_group(f'capsule/{capsule_name}')
        for chunk in all_chunks:
            hdf5_key = _chunk_id_to_hdf5_key(chunk['file'], chunk['chunk_name'])
            chunk_id  = generate_chunk_id(chunk['file'], chunk['chunk_name'])
            ds = capsule_group.create_dataset(
                hdf5_key,
                data=chunk['code'].encode('utf-8'),
                dtype=str_dtype,
            )
            # Store the URI so readers can join against the JSONL embeddings file.
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

    # Normalise separators
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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()

    print(f'Starting chunking for capsule: {CAPSULE_NAME}')
    print(f'Capsule path:     {CAPSULE_PATH}')
    print(f'Output path:      {OUTPUT_PATH}')
    print(f'Excluded folders: {", ".join(EXCLUDED_FOLDERS)}')
    print(f'Max file chars:   {MAX_FILE_CHARS}')
    print('---')

    ts_files = find_ts_files(CAPSULE_PATH)
    print(f'Found {len(ts_files)} TypeScript files to process')
    print('---')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = SCRIPT_DIR.parent / 'output' / f'{CAPSULE_NAME.lower()}-embeddings.jsonl'
    hdf5_path  = SCRIPT_DIR.parent / 'output' / f'{CAPSULE_NAME.lower()}-code.h5'

    # Clear outputs
    OUTPUT_PATH.write_text('')
    jsonl_path.write_text('')
    if hdf5_path.exists():
        hdf5_path.unlink()

    all_chunks: list[dict] = []
    errors: list[str] = []

    for file_path in tqdm(ts_files, desc='Processing files',
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        try:
            relative_path = str(Path(file_path).relative_to(CAPSULE_PATH.parent))
        except ValueError:
            relative_path = file_path

        tqdm.write(f'\nProcessing: {relative_path}')
        file_t0 = time.time()

        try:
            response, parsed_chunks = process_file(file_path)

            with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                f.write(response + '\n\n')

            if parsed_chunks:
                all_chunks.extend(parsed_chunks)
                append_jsonl_output(parsed_chunks, jsonl_path)
                tqdm.write(f'  ✓ {len(parsed_chunks)} chunk(s) in {time.time() - file_t0:.1f}s')
            else:
                tqdm.write(f'  → No chunks extracted ({time.time() - file_t0:.1f}s)')

        except Exception as e:
            tqdm.write(f'  ✗ Fatal error: {e}')
            errors.append(relative_path)
            with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                f.write(f'## ERROR processing {relative_path}: {e}\n\n')

    tqdm.write(f'\nWriting HDF5 with {len(all_chunks)} chunks...')
    if all_chunks:
        write_hdf5_output(all_chunks, hdf5_path, CAPSULE_NAME)

    post_process_output(OUTPUT_PATH)

    elapsed = time.time() - start_time
    print('---')
    print(f'Done in {elapsed:.1f}s')
    print(f'Total chunks:  {len(all_chunks)}')
    print(f'Failed files:  {len(errors)}')
    if errors:
        for e in errors:
            print(f'  ✗ {e}')
    print(f'Outputs:')
    print(f'  MD (backup):        {OUTPUT_PATH}')
    print(f'  JSONL (embeddings): {jsonl_path}')
    print(f'  HDF5 (code):        {hdf5_path}')


if __name__ == '__main__':
    main()
