# Phase 2: RAG Query Pipeline — Implementation Plan

## What We're Building

A single script (`rag_query.py`) that does:
```
User question → Hypencoder retrieval (q-net) → Top-K docs → LLM → Answer
```

## Your Setup (Confirmed)

| Item | Value |
|------|-------|
| Model | `jfkback/hypencoder.2_layer` |
| Encoded docs path | `cache/encoded_docs` *(you'll fill in the real path)* |
| Doc count | 120,036 |
| Embedding dim | 768 |
| Device | CPU |

---

## Step 1: Fix `retrieve.py` broken import

### Problem

[retrieve.py](/hypencoder_cb/inference/retrieve.py) has these imports at the top (lines 18-26):

```python
from hypencoder_cb.utils.data_utils import (
    load_qrels_from_ir_datasets,
    load_qrels_from_json,
)
from hypencoder_cb.utils.eval_utils import (
    calculate_metrics_to_file,
    load_standard_format_as_run,
    pretty_print_standard_format,
)
```

These crash on import because `data_utils.py` depends on `ir_datasets` which you may not have, and `eval_utils.py` tries to import `ir_measures`. We only need the `HypencoderRetriever` class — not evaluation.

### Fix

Move those imports inside `do_eval_and_pretty_print()` (the only function that uses them). Change lines 18-26 from top-level imports to lazy imports inside the function.

#### [MODIFY] [retrieve.py](/hypencoder-paper/hypencoder_cb/inference/retrieve.py)

**Remove** lines 18-26 (the `data_utils` and `eval_utils` imports).

**Add** them as lazy imports inside `do_eval_and_pretty_print()` at line 183:

```python
def do_eval_and_pretty_print(
    retrieval_path: str,
    output_dir: str,
    ir_dataset_name: Optional[str] = None,
    qrel_json: Optional[str] = None,
    metric_names: Optional[List[str]] = None,
) -> None:
    # Lazy imports — only needed for evaluation, not retrieval
    from hypencoder_cb.utils.data_utils import load_qrels_from_ir_datasets, load_qrels_from_json
    from hypencoder_cb.utils.eval_utils import (
        calculate_metrics_to_file,
        load_standard_format_as_run,
        pretty_print_standard_format,
    )
    ...  # rest of function unchanged
```

---

## Step 2: Create `rag_query.py`

#### [NEW] [rag_query.py](/hypencoder-paper/rag_query.py)

This is the only new file. Complete code below:

```python
"""
rag_query.py — Interactive RAG pipeline using Hypencoder retrieval + LLM.

Usage:
    # Retrieval only (test that retrieval works):
    python rag_query.py --no-llm

    # Full RAG (retrieval + LLM):
    python rag_query.py
"""

import argparse
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_cb.inference.shared import (
    Item,
    TextQuery,
    load_encoded_items_from_disk,
)
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.utils.iterator_utils import batchify_slicing
from hypencoder_cb.utils.torch_utils import dtype_lookup


# ─── CONFIGURATION (fill in your paths) ─────────────────────────────────────

MODEL_PATH = "jfkback/hypencoder.2_layer"          # Must match what you encoded with
ENCODED_DOCS_PATH = "cache/encoded_docs"            # <-- UPDATE THIS to your real path
TOP_K = 5                                           # Number of chunks to retrieve
QUERY_MAX_LENGTH = 64                               # Max tokens for query
BATCH_SIZE = 50_000                                 # Scoring batch size (tuned for CPU)

# ─── LLM INTEGRATION (placeholder for Samsung) ──────────────────────────────

def call_llm(prompt: str) -> str:
    """
    Placeholder for Samsung's LLM API.

    Replace this function with your actual LLM call, e.g.:
        import requests
        response = requests.post("https://your-samsung-llm-endpoint/v1/chat", json={...})
        return response.json()["answer"]
    """
    return (
        "[LLM placeholder] Replace call_llm() in rag_query.py with your Samsung LLM API.\n"
        f"The prompt that would be sent ({len(prompt)} chars) starts with:\n"
        f"{prompt[:300]}..."
    )


# ─── RAG PROMPT TEMPLATE ────────────────────────────────────────────────────

def build_rag_prompt(query: str, retrieved_docs: list[Item]) -> str:
    """Build a RAG prompt with retrieved context."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"--- Document {i} (ID: {doc.id}, Score: {doc.score:.4f}) ---\n{doc.text}")

    context = "\n\n".join(context_parts)

    return f"""You are a helpful assistant that answers questions about a Bixby capsules/agents codebase.
Use ONLY the provided context documents to answer. If the context doesn't contain enough information, say so.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


# ─── RETRIEVER ───────────────────────────────────────────────────────────────

class SimpleHypencoderRetriever:
    """
    Simplified retriever that loads the model + embeddings once,
    then answers queries interactively.
    """

    def __init__(self, model_path: str, encoded_docs_path: str, device: str = "cpu"):
        print(f"\n{'='*60}")
        print(f"Loading Hypencoder model: {model_path}")
        print(f"{'='*60}")

        self.device = device
        self.dtype = torch.float32

        # 1. Load the dual encoder model (query encoder + passage encoder)
        self.model = (
            HypencoderDualEncoder.from_pretrained(model_path)
            .to(device, dtype=self.dtype)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Model loaded")

        # 2. Load pre-encoded document embeddings from Phase 1
        print(f"\nLoading encoded documents from: {encoded_docs_path}")
        encoded_items = list(load_encoded_items_from_disk(encoded_docs_path))

        self.doc_ids = [item.id for item in encoded_items]
        self.doc_texts = [item.text for item in encoded_items]
        self.doc_embeddings = torch.stack([
            torch.tensor(item.representation, dtype=self.dtype)
            for item in tqdm(encoded_items, desc="Stacking embeddings")
        ])

        print(f"✓ Loaded {len(self.doc_ids)} documents")
        print(f"  Embedding shape: {self.doc_embeddings.shape}")
        print(f"  Memory: ~{self.doc_embeddings.nelement() * 4 / 1024 / 1024:.0f} MB")

    def retrieve(self, query_text: str, top_k: int = TOP_K) -> list[Item]:
        """
        Take a query string, generate a q-net, score all docs, return top-k.

        What happens under the hood:
        1. Tokenize query text
        2. Pass through BERT → get hidden states
        3. Hyper-head reads hidden states → generates weights/biases for a 2-layer MLP
        4. That MLP (the q-net) scores every document embedding
        5. Return top-k highest scoring documents
        """

        # Step 1: Tokenize the query
        tokenized = self.tokenizer(
            query_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=QUERY_MAX_LENGTH,
        ).to(self.device)

        # Step 2+3: Generate the q-net for this specific query
        with torch.no_grad():
            query_output = self.model.query_encoder(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )
        q_net = query_output.representation  # This is a callable neural network!

        # Step 4: Score all documents in batches
        all_scores = []
        for batch_embeddings in batchify_slicing(self.doc_embeddings, BATCH_SIZE):
            batch_embeddings = batch_embeddings.unsqueeze(0)  # Shape: (1, batch_size, 768)
            scores = q_net(batch_embeddings).squeeze()         # Shape: (batch_size,)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores)

        # Step 5: Get top-k
        values, indices = torch.topk(all_scores, top_k)

        results = []
        for score, idx in zip(values, indices):
            results.append(Item(
                text=self.doc_texts[idx],
                id=self.doc_ids[idx],
                score=score.item(),
            ))

        return results


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hypencoder RAG Query Tool")
    parser.add_argument("--no-llm", action="store_true", help="Retrieval only, skip LLM call")
    parser.add_argument("--model", default=MODEL_PATH, help="Model path or HF repo")
    parser.add_argument("--docs", default=ENCODED_DOCS_PATH, help="Path to encoded docs")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of docs to retrieve")
    args = parser.parse_args()

    # Load everything once
    retriever = SimpleHypencoderRetriever(
        model_path=args.model,
        encoded_docs_path=args.docs,
    )

    print(f"\n{'='*60}")
    print("Ready! Type your question (or 'quit' to exit)")
    if args.no_llm:
        print("Mode: RETRIEVAL ONLY (--no-llm)")
    else:
        print("Mode: FULL RAG (retrieval + LLM)")
    print(f"{'='*60}\n")

    while True:
        try:
            query = input("🔍 Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Retrieve
        print(f"\nSearching {len(retriever.doc_ids)} documents...")
        results = retriever.retrieve(query, top_k=args.top_k)

        # Show retrieved chunks
        print(f"\n{'─'*60}")
        print(f"Top {len(results)} retrieved documents:")
        print(f"{'─'*60}")
        for i, doc in enumerate(results, 1):
            text_preview = doc.text[:200].replace('\n', ' ')
            print(f"\n  [{i}] Score: {doc.score:.4f}")
            print(f"      ID: {doc.id}")
            print(f"      Text: {text_preview}...")

        # LLM call (if enabled)
        if not args.no_llm:
            prompt = build_rag_prompt(query, results)
            print(f"\n{'─'*60}")
            print("LLM Answer:")
            print(f"{'─'*60}")
            answer = call_llm(prompt)
            print(f"\n{answer}")

        print()


if __name__ == "__main__":
    main()
```

---

## Step-by-Step Execution Plan

After you approve, I will:

| Step | Action | File |
|------|--------|------|
| 1 | Move `data_utils` and `eval_utils` imports inside `do_eval_and_pretty_print()` | [retrieve.py](/hypencoder_cb/inference/retrieve.py) |
| 2 | Create `rag_query.py` with the complete code above | [rag_query.py](/rag_query.py) |
| 3 | Test retrieval with `python rag_query.py --no-llm --docs YOUR_PATH` | Terminal |

---

## After This

1. **You update `ENCODED_DOCS_PATH`** in the script to point to your real `cache/encoded_docs` folder
2. **Test retrieval** with `--no-llm` flag to see if the right chunks come back
3. **Wire up Samsung LLM** — replace the `call_llm()` function with your actual API call
4. **Run full RAG** without the `--no-llm` flag

## Verification

- Run `python rag_query.py --no-llm` → should load model, load 120k embeddings, accept a query, print top-5 docs with scores
- Confirm retrieved docs are relevant to the question
- Check scores are reasonable (not all identical)
