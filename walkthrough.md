# Phase 2 Handoff — Hypencoder RAG Pipeline

> Give this to the AI agent in your other repo. It has everything needed.

## Prerequisites

- Same `hypencoder_cb/` folder present in the repo
- Encoded docs from Phase 1 (120k docs, encoded with `jfkback/hypencoder.2_layer`)
- Python environment with: `torch`, `transformers`, `tqdm`, `docarray`, `fire`, `numpy`

---

## Change 1: Fix `hypencoder_cb/inference/retrieve.py`

Apply this diff — it fixes a crash-on-import bug and a performance issue:

```diff
--- a/hypencoder_cb/inference/retrieve.py
+++ b/hypencoder_cb/inference/retrieve.py
@@ -17,12 +17,6 @@
 )
 from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
-from hypencoder_cb.utils.data_utils import (
-    load_qrels_from_ir_datasets,
-    load_qrels_from_json,
-)
-from hypencoder_cb.utils.eval_utils import (
-    calculate_metrics_to_file,
-    load_standard_format_as_run,
-    pretty_print_standard_format,
-)
 from hypencoder_cb.utils.iterator_utils import batchify_slicing
 from hypencoder_cb.utils.torch_utils import dtype_lookup
```

Then in `__init__` of `HypencoderRetriever`, change the loading block from:

```python
        print("Started loading encoded items...")
        encoded_items = load_encoded_items_from_disk(
            encoded_item_path,
        )
        # ... (torch.stack as before) ...
        self.encoded_item_ids = [x.id for x in tqdm(encoded_items)]
        self.encoded_item_texts = [x.text for x in tqdm(encoded_items)]
```

To:

```python
        print("Started loading encoded items...")
        encoded_items = list(load_encoded_items_from_disk(
            encoded_item_path,
        ))
        print(f"Building tensors for {len(encoded_items)} items...")
        # ... (torch.stack as before) ...
        self.encoded_item_ids = [x.id for x in encoded_items]
        self.encoded_item_texts = [x.text for x in encoded_items]
```

And add lazy imports at the top of `do_eval_and_pretty_print()`:

```python
def do_eval_and_pretty_print(...):
    # ADD THESE as the first lines of the function body:
    from hypencoder_cb.utils.data_utils import (
        load_qrels_from_ir_datasets,
        load_qrels_from_json,
    )
    from hypencoder_cb.utils.eval_utils import (
        calculate_metrics_to_file,
        load_standard_format_as_run,
        pretty_print_standard_format,
    )

    # rest of function unchanged...
```

---

## Change 2: Create `rag_query.py` at project root

```python
"""
rag_query.py — Interactive RAG: Hypencoder retrieval + LLM answer

Usage:
    python rag_query.py                   # full RAG (retrieval + LLM)
    python rag_query.py --no-llm          # retrieval only (for testing)
    python rag_query.py --top-k 10        # retrieve 10 chunks instead of 5
"""

import argparse

from hypencoder_cb.inference.retrieve import HypencoderRetriever
from hypencoder_cb.inference.shared import TextQuery, Item
from typing import List

# ── CONFIG — update ENCODED_DOCS_PATH to your real path ───────────────────────
MODEL_PATH        = "jfkback/hypencoder.2_layer"
ENCODED_DOCS_PATH = "cache/encoded_docs"   # <── UPDATE THIS to your actual path
TOP_K             = 5
QUERY_MAX_LENGTH  = 64
BATCH_SIZE        = 50_000   # tuned for CPU with 120k docs


# ── LLM — replace with Samsung's API ──────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """
    Plug in Samsung's LLM API here.

    Example skeleton:
        import requests
        r = requests.post("https://your-llm-endpoint/chat", json={"prompt": prompt})
        return r.json()["text"]
    """
    return (
        "[LLM not connected] Replace call_llm() in rag_query.py with your Samsung LLM API.\n"
        f"Prompt ({len(prompt)} chars):\n{prompt[:500]}..."
    )


# ── RAG PROMPT ─────────────────────────────────────────────────────────────────

def build_prompt(query: str, docs: List[Item]) -> str:
    """Build a RAG prompt from retrieved context documents."""
    context = "\n\n".join(
        f"[{i+1}] (id={doc.id}, score={doc.score:.4f})\n{doc.text}"
        for i, doc in enumerate(docs)
    )
    return (
        "You are a helpful assistant for a Bixby capsules/agents codebase.\n"
        "Answer using ONLY the context below. If the context doesn't contain "
        "enough information, say so.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hypencoder RAG Query Tool")
    parser.add_argument("--no-llm",  action="store_true", help="Retrieval only, skip LLM")
    parser.add_argument("--model",   default=MODEL_PATH,  help="Model path or HF repo")
    parser.add_argument("--docs",    default=ENCODED_DOCS_PATH, help="Path to encoded docs")
    parser.add_argument("--top-k",   type=int, default=TOP_K, help="Number of docs to retrieve")
    args = parser.parse_args()

    # Load model + embeddings once (takes ~30-60 seconds on CPU for 120k docs)
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model}")
    print(f"Loading docs:  {args.docs}")
    print(f"{'='*60}\n")

    retriever = HypencoderRetriever(
        model_name_or_path=args.model,
        encoded_item_path=args.docs,
        device="cpu",                       # explicit CPU — default is "cuda"
        dtype="float32",
        query_max_length=QUERY_MAX_LENGTH,
        batch_size=BATCH_SIZE,
        put_all_embeddings_on_device=True,  # keep in RAM (fine for CPU)
    )

    num_docs = len(retriever.encoded_item_ids)
    print(f"\n{'='*60}")
    print(f"Ready! {num_docs} docs loaded.")
    print(f"Embedding shape: {retriever.encoded_item_embeddings.shape}")
    if args.no_llm:
        print("Mode: RETRIEVAL ONLY (--no-llm)")
    else:
        print("Mode: FULL RAG (retrieval + LLM)")
    print(f"Type your question, or 'quit' to exit.")
    print(f"{'='*60}\n")

    while True:
        try:
            query = input("Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Retrieve top-k using Hypencoder q-net scoring
        results = retriever.retrieve(
            TextQuery(text=query, id="interactive"),
            top_k=args.top_k,
        )

        # Show retrieved chunks
        print(f"\n{'─'*60}")
        print(f"Top {len(results)} retrieved chunks:")
        print(f"{'─'*60}")
        for i, doc in enumerate(results, 1):
            preview = doc.text[:200].replace('\n', ' ')
            print(f"\n  [{i}] score={doc.score:.4f}  id={doc.id}")
            print(f"       {preview}...")

        # LLM answer (if enabled)
        if not args.no_llm:
            prompt = build_prompt(query, results)
            print(f"\n{'─'*60}")
            print("LLM Answer:")
            print(f"{'─'*60}")
            print(call_llm(prompt))

        print()


if __name__ == "__main__":
    main()
```

---

## Critical Notes

| Rule | Why |
|------|-----|
| **Model MUST be `jfkback/hypencoder.2_layer`** | Docs were encoded with this model. Using a different model = garbage scores. |
| **Device MUST be `"cpu"`** | No GPU available. Default is `"cuda"` which will crash. |
| **`ENCODED_DOCS_PATH` must use same format as encoding** | If you encoded with a relative path like `cache/encoded_docs`, use the same relative path here. Windows absolute paths with `C:\` may break the `file://` URI in docarray. |
| **`BATCH_SIZE = 50_000`** | Splits 120k docs into 3 scoring batches. Keeps memory manageable on CPU. |

---

## Test Steps

```bash
# 1. Test the import fix works
python -c "from hypencoder_cb.inference.retrieve import HypencoderRetriever; print('OK')"

# 2. Test retrieval only (no LLM needed)
python rag_query.py --no-llm --docs "cache/encoded_docs"

# 3. Expected output:
#    - Loads model (~10 sec)
#    - Loads 120,036 embeddings (~30 sec)
#    - Accepts a query
#    - Returns 5 results with scores and doc IDs
#    - Each result shows a text preview from your Bixby corpus

# 4. Once retrieval works, wire up Samsung LLM in call_llm() and run:
python rag_query.py --docs "cache/encoded_docs"
```
