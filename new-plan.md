# TypeScript Codebase RAG Pipeline v3 — Implementation Plan
### TS-Only · LanceDB · SQLite · Hybrid BM25+Vector · Graph Expansion · Reranking

---

## How to Read This Plan

Each phase is self-contained. Complete phases in order — later phases import from earlier ones. Every phase lists the exact files to create, the key interfaces/types to define, the logic to implement, and a short acceptance checklist an AI coder can verify before moving on.

**LLM answer generation is intentionally out of scope.** A `PlaceholderLlmProvider` is wired in at Phase 8; swap it with a real provider later. Everything else — extraction, graph, storage, indexing, retrieval — is fully implemented.

---

## Dependency Map

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
                                       │
                                    Phase 5 ──► Phase 6
                                       │
                                    Phase 7 ──► Phase 8
```

---

## Dependency Manifest

Install all of these once at project root before starting Phase 1.

```json
{
  "dependencies": {
    "ts-morph": "^22.0.0",
    "better-sqlite3": "^9.6.0",
    "@lancedb/lancedb": "^0.12.0",
    "apache-arrow": "^14.0.2",
    "p-limit": "^5.0.0",
    "zod": "^3.23.0",
    "commander": "^12.1.0",
    "glob": "^11.0.0",
    "openai": "^4.52.0"
  },
  "devDependencies": {
    "typescript": "^5.5.0",
    "@types/better-sqlite3": "^7.6.11",
    "@types/node": "^20.0.0",
    "tsx": "^4.16.0",
    "vitest": "^2.0.0"
  }
}
```

> **Note on embeddings:** `openai` is used as the embedding provider (calls `text-embedding-3-small`). The `EmbeddingProvider` interface is pluggable — swap with `@xenova/transformers` or any other provider without touching retrieval code.

---

## Folder Structure (complete)

```
src/
├── config/
│   ├── schema.ts            # Zod schema + Config type
│   └── defaults.ts          # Default config values
├── types/
│   └── index.ts             # All shared domain types
├── extraction/
│   └── ts/
│       ├── TsExtractor.ts   # ts-morph symbol extraction
│       ├── EdgeExtractor.ts # import/call/inheritance edges
│       └── Chunker.ts       # sliding-window chunker
├── graph/
│   ├── EdgeKind.ts          # EdgeKind enum + default weights
│   ├── Graph.ts             # in-memory Graph class
│   ├── TestLinker.ts        # test-file ↔ source-function linker (3 signals)
│   └── GraphBuilder.ts      # orchestrates extraction → graph
├── storage/
│   ├── sqlite/
│   │   ├── schema.sql       # DDL
│   │   ├── migrations.ts    # migration runner
│   │   └── SqliteStore.ts   # CRUD over nodes/edges/chunks/BM25
│   └── lancedb/
│       ├── LanceSchema.ts   # Arrow schema definition
│       └── LanceStore.ts    # upsert + vector search
├── embedding/
│   ├── EmbeddingProvider.ts       # interface
│   ├── OpenAiEmbeddingProvider.ts # default implementation
│   ├── EmbeddingTextBuilder.ts    # builds rich embedding text per chunk
│   └── EmbeddingPipeline.ts       # bounded-concurrency batch embed+upsert
├── search/
│   ├── bm25/
│   │   ├── Tokenizer.ts        # lowercase + alphanum tokenizer
│   │   ├── InvertedIndex.ts    # in-memory inverted index + persistence
│   │   └── Bm25Scorer.ts       # BM25 query scorer
│   ├── vector/
│   │   └── VectorSearcher.ts   # wraps LanceStore.search()
│   └── hybrid/
│       ├── RrfFusion.ts        # Reciprocal Rank Fusion
│       └── HybridRetriever.ts  # BM25 + vector + RRF
├── expansion/
│   ├── BudgetConfig.ts         # expansion budget types
│   └── GraphExpander.ts        # bounded graph expansion
├── query/
│   ├── ContextAssembler.ts     # assembles chunks + graph context string
│   ├── QueryEngine.ts          # public query() entry point
│   └── llm/
│       ├── LlmProvider.ts               # interface
│       └── PlaceholderLlmProvider.ts    # returns context only, no LLM call
├── indexer/
│   └── Indexer.ts              # orchestrates full index build
└── cli/
    └── index.ts                # commander CLI: index | query | inspect
tests/
├── extraction/
├── graph/
├── search/
└── query/
```

---

## Phase 1 — Project Scaffold & Shared Type System

**Goal:** All TypeScript compiles. Every downstream type is defined here. No logic yet.

### Files to Create

#### `tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "declaration": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

#### `src/types/index.ts`

Define all domain types. No implementation here — only type/interface/enum declarations.

```typescript
// ─── Node kinds ──────────────────────────────────────────────────────────────

export type NodeKind =
  | 'TS_FUNCTION'
  | 'TS_CLASS'
  | 'TS_INTERFACE'
  | 'TS_TYPE_ALIAS'
  | 'TS_VARIABLE'
  | 'TS_MODULE';

// ─── Path metadata (on every node and every chunk) ───────────────────────────

export interface PathMeta {
  absolutePath: string;
  relativePath: string;        // relative to codebaseRoot
  fileName: string;
  extension: string;
  folderHierarchy: string[];   // e.g. ['src', 'extraction', 'ts']
}

// ─── Symbol node ─────────────────────────────────────────────────────────────

export interface SymbolNode {
  id: string;                  // sha1(relativePath + ':' + qualifiedName + ':' + startLine)
  kind: NodeKind;
  name: string;
  qualifiedName: string;       // e.g. 'MyClass.myMethod'
  signature?: string;          // full TypeScript signature string
  jsDoc?: string;
  startLine: number;
  endLine: number;
  pathMeta: PathMeta;
  hubDegree?: number;          // set by GraphBuilder after all edges are collected
}

// ─── Graph edge ───────────────────────────────────────────────────────────────

export type EdgeKind =
  | 'IMPORTS'
  | 'CALLS'
  | 'EXTENDS'
  | 'IMPLEMENTS'
  | 'OVERRIDES'
  | 'TESTS'       // test function/describe → source function
  | 'TESTED_BY';  // source function → test function (reverse)

export interface GraphEdge {
  id: string;     // sha1(fromId + '|' + toId + '|' + kind)
  fromId: string;
  toId: string;
  kind: EdgeKind;
  weight?: number;
  meta?: Record<string, string>;
}

// ─── Chunk ────────────────────────────────────────────────────────────────────

export interface ChunkRecord {
  id: string;             // sha1(nodeId + ':' + chunkIndex)
  nodeId: string;
  chunkIndex: number;
  totalChunks: number;
  text: string;           // raw source text of this chunk
  embeddingText: string;  // enriched text used for embedding
  pathMeta: PathMeta;
  startLine: number;
  endLine: number;
  tokenCount: number;
  contentHash: string;    // sha1(text) — used for incremental updates
}

// ─── BM25 ────────────────────────────────────────────────────────────────────

export interface BM25Posting {
  chunkId: string;
  tf: number;
}

export interface BM25IndexEntry {
  term: string;
  idf: number;
  postings: BM25Posting[];
}

// ─── Retrieval ────────────────────────────────────────────────────────────────

export interface SearchHit {
  chunkId: string;
  nodeId: string;
  score: number;
  chunk: ChunkRecord;
  source: 'bm25' | 'vector' | 'hybrid';
}

export interface GraphContext {
  seedNodeIds: string[];
  visitedNodeIds: string[];
  edges: GraphEdge[];
  nodes: SymbolNode[];
}

export interface QueryResult {
  query: string;
  hits: SearchHit[];
  graphContext: GraphContext;
  assembledContext: string;
  llmAnswer: string | null;  // null when using PlaceholderLlmProvider
}
```

#### `src/config/schema.ts`

```typescript
import { z } from 'zod';

const EdgeKindWeightsSchema = z.object({
  IMPORTS:    z.number().default(0.5),
  CALLS:      z.number().default(1.0),
  EXTENDS:    z.number().default(0.9),
  IMPLEMENTS: z.number().default(0.9),
  OVERRIDES:  z.number().default(0.8),
  TESTS:      z.number().default(1.2),
  TESTED_BY:  z.number().default(1.2),
});

export const ConfigSchema = z.object({
  codebaseRoot: z.string(),
  outputDir: z.string().default('.rag-cache'),

  // Extraction
  tsConfigPath: z.string().optional(),
  includeGlobs: z.array(z.string()).default(['**/*.ts', '**/*.tsx']),
  excludeGlobs: z.array(z.string()).default([
    '**/node_modules/**', '**/dist/**', '**/*.d.ts'
  ]),

  // Chunking
  chunkMaxTokens: z.number().default(400),
  chunkOverlapTokens: z.number().default(80),

  // Embedding
  embeddingModel: z.string().default('text-embedding-3-small'),
  embeddingDimensions: z.number().default(1536),
  embeddingConcurrency: z.number().default(8),

  // Graph expansion budget
  expansionBudget: z.object({
    seedCap: z.number().default(10),
    hubDegreeCap: z.number().default(50),
    maxVisitedNodes: z.number().default(60),
    edgeKindWeights: EdgeKindWeightsSchema.default({}),
    excludedEdgeKinds: z.array(z.string()).default([]),
  }).default({}),

  // Retrieval
  bm25TopK: z.number().default(20),
  vectorTopK: z.number().default(20),
  hybridTopK: z.number().default(10),
  rrfK: z.number().default(60),
});

export type Config = z.infer<typeof ConfigSchema>;
```

#### `src/config/defaults.ts`

```typescript
import type { Config } from './schema.js';

export const DEFAULT_CONFIG: Partial<Config> = {
  outputDir: '.rag-cache',
  includeGlobs: ['**/*.ts', '**/*.tsx'],
  excludeGlobs: ['**/node_modules/**', '**/dist/**', '**/*.d.ts'],
  chunkMaxTokens: 400,
  chunkOverlapTokens: 80,
  embeddingModel: 'text-embedding-3-small',
  embeddingDimensions: 1536,
  embeddingConcurrency: 8,
};
```

### Acceptance Checklist
- [ ] `npx tsx src/types/index.ts` exits without error
- [ ] `ConfigSchema.parse({ codebaseRoot: '.' })` returns a valid config object with all defaults populated
- [ ] `npx tsc --noEmit` reports zero errors

---

## Phase 2 — TypeScript Extraction Layer

**Goal:** Walk the TypeScript project with ts-morph. Emit `SymbolNode[]`, `GraphEdge[]`, and raw `ChunkRecord[]` (without embedding text yet — `embeddingText` is set to `''` as placeholder).

### Files to Create

#### `src/extraction/ts/TsExtractor.ts`

**Implements:** `extract(config: Config): Promise<{ nodes: SymbolNode[], rawChunks: Omit<ChunkRecord, 'embeddingText'>[], project: Project }>`

Logic:
1. Create a `ts-morph` `Project`. If `config.tsConfigPath` is set, pass it to `addSourceFilesFromTsConfig`. Otherwise, call `addSourceFilesByGlob(config.includeGlobs)` from `config.codebaseRoot`. Apply `excludeGlobs` via `Project` options.
2. For each source file, extract the following symbol types:
   - **Functions** via `sourceFile.getFunctions()` — capture `name`, full signature string (parameter types + return type via `getSignature().getDeclaration()`), JSDoc text, `getStartLineNumber()`, `getEndLineNumber()`
   - **Classes** via `sourceFile.getClasses()` — capture class-level info, then recurse into `cls.getMethods()` for each method
   - **Interfaces** via `sourceFile.getInterfaces()`
   - **Type aliases** via `sourceFile.getTypeAliases()`
   - **Exported variables** via `sourceFile.getVariableDeclarations().filter(v => v.isExported())`
3. Build `PathMeta` for each file:
   - `absolutePath`: `sourceFile.getFilePath()`
   - `relativePath`: `path.relative(config.codebaseRoot, absolutePath)`
   - `fileName`: `path.basename(absolutePath)`
   - `extension`: `path.extname(absolutePath)`
   - `folderHierarchy`: `relativePath.split('/').slice(0, -1)`
4. Assign a stable `id` to each node: `crypto.createHash('sha1').update(relativePath + ':' + qualifiedName + ':' + startLine).digest('hex')`
5. Pass each node + its source text to `Chunker.chunkNode()` and accumulate raw chunks.
6. Return `{ nodes, rawChunks, project }` — `project` is needed by `EdgeExtractor`.

#### `src/extraction/ts/EdgeExtractor.ts`

**Implements:** `extractEdges(project: Project, nodes: SymbolNode[]): GraphEdge[]`

Logic:
1. Build a lookup map `nodeByQualified: Map<string, string>` from `qualifiedName → nodeId`.
2. Also build `nodeByFile: Map<string, string[]>` from `relativePath → nodeId[]` for import resolution.
3. For each source file:
   - **IMPORTS:** For each `ImportDeclaration`, resolve the module specifier to an absolute path. If the resolved path belongs to a file in the project, emit `IMPORTS` edges from each importing symbol (or the file's module node) to each named imported symbol.
   - **CALLS:** For each `CallExpression` found by `sourceFile.getDescendantsOfKind(SyntaxKind.CallExpression)`, get the symbol of the callee via `expr.getExpression().getSymbol()`. If it resolves to a node in the map, find the nearest ancestor function/method node and emit a `CALLS` edge.
   - **EXTENDS:** For each class, `cls.getBaseClass()` — if found in the map, emit `EXTENDS`.
   - **IMPLEMENTS:** For each class, `cls.getImplements()` — emit `IMPLEMENTS` for each.
   - **OVERRIDES:** For each method where `method.isOverride()` is true, find the base class method and emit `OVERRIDES`.
4. Compute `id = sha1(fromId + '|' + toId + '|' + kind)` per edge. Deduplicate by `id`.

#### `src/extraction/ts/Chunker.ts`

**Implements:** `chunkNode(node: SymbolNode, sourceText: string, config: Config): Omit<ChunkRecord, 'embeddingText'>[]`

Logic (sliding window by approximate token count):
1. Approximate token count: `Math.ceil(words.length * 1.3)` where `words = sourceText.split(/\s+/)`.
2. If it fits within `config.chunkMaxTokens`, return a single chunk.
3. Otherwise, split into overlapping windows:
   - Window size in words: `Math.floor(config.chunkMaxTokens / 1.3)`
   - Stride in words: `Math.floor((config.chunkMaxTokens - config.chunkOverlapTokens) / 1.3)`
4. For each window: assign `chunkIndex`, compute `startLine` / `endLine` by counting newlines in the text before the window offset, set `contentHash = sha1(windowText)`.
5. Set `totalChunks` on all chunks from this node after the full split is known.
6. Leave `embeddingText: ''` — filled in by `EmbeddingTextBuilder` in Phase 5.

### Acceptance Checklist
- [ ] Running `TsExtractor.extract()` on the project's own `src/` directory produces > 0 nodes and > 0 chunks
- [ ] Every node has a non-empty `id`, `name`, `pathMeta.relativePath`, and correct `startLine`
- [ ] `EdgeExtractor` produces `CALLS` edges when one function in `src/` calls another
- [ ] A class with two methods produces 1 `TS_CLASS` node and 2 `TS_FUNCTION` nodes
- [ ] No chunk has `tokenCount` exceeding `chunkMaxTokens * 1.1` (10% slack for approximation)

---

## Phase 3 — Graph Construction & Test Linker

**Goal:** Build a unified in-memory graph from Phase 2 output. Add `TESTS` / `TESTED_BY` edges by linking `.test.ts` / `.spec.ts` files to the source functions they test via a three-signal linker. Compute `hubDegree` on all nodes.

### Files to Create

#### `src/graph/EdgeKind.ts`

```typescript
export const EDGE_KIND_DEFAULT_WEIGHTS: Record<string, number> = {
  IMPORTS:    0.5,
  CALLS:      1.0,
  EXTENDS:    0.9,
  IMPLEMENTS: 0.9,
  OVERRIDES:  0.8,
  TESTS:      1.2,
  TESTED_BY:  1.2,
};
```

#### `src/graph/Graph.ts`

```typescript
import type { SymbolNode, GraphEdge } from '../types/index.js';

export class Graph {
  private nodes   = new Map<string, SymbolNode>();
  private edges   = new Map<string, GraphEdge>();
  private outEdges = new Map<string, Set<string>>(); // nodeId → Set<edgeId>
  private inEdges  = new Map<string, Set<string>>();

  addNode(node: SymbolNode): void;
  addEdge(edge: GraphEdge): void;
  getNode(id: string): SymbolNode | undefined;
  getEdge(id: string): GraphEdge | undefined;
  getOutEdges(nodeId: string): GraphEdge[];
  getInEdges(nodeId: string): GraphEdge[];
  getDegree(nodeId: string): number;     // in-degree + out-degree
  allNodes(): SymbolNode[];
  allEdges(): GraphEdge[];

  // Sets node.hubDegree = getDegree(node.id) for every node in the graph.
  computeHubDegrees(): void;
}
```

#### `src/graph/TestLinker.ts`

**Implements:** `link(nodes: SymbolNode[], existingEdges: GraphEdge[]): GraphEdge[]`

Detects which test functions/describes target which source functions using three signals. Returns only new edges (does not duplicate edges already in `existingEdges`).

**Signal 1 — File name proximity**

For every node where `pathMeta.fileName` matches `*.test.ts`, `*.spec.ts`, `*.test.tsx`, or `*.spec.tsx`:
- Derive the likely source file name by stripping `.test` / `.spec` (e.g. `payment.test.ts` → `payment.ts`).
- Find all nodes whose `pathMeta.fileName === derivedSourceName`.
- Emit `TESTS` edges from each function node in the test file to each function node in the source file. Set `meta.signal = '1'`.

**Signal 2 — Import path**

For every `IMPORTS` edge already in `existingEdges` where the `fromId` node is in a test file (`.test.ts` / `.spec.ts`) and the `toId` node is in a non-test file:
- Emit `TESTS` edge from the importing test function to the imported source function.
- Set `meta.signal = '2'`.

**Signal 3 — Name mention in test body**

For every function node in a test file, read its `name`. Look for source-file function nodes whose `name` appears as a substring inside the test function's `name` (e.g. test function `it_should_call_handlePayment` mentions `handlePayment`). If the normalized names overlap:
- Compute match confidence: exact match = 1.0, camelCase sub-match = 0.7, lowercase sub-match = 0.5. Only emit if confidence ≥ 0.7.
- Emit `TESTS` edge. Set `meta.signal = '3'`.

**De-duplication:** Use edge `id` (sha1 of `fromId|toId|kind`) to prevent duplicates across signals or with existing edges.

**Reverse edges:** For every `TESTS` edge emitted, also emit the corresponding `TESTED_BY` edge (swap `fromId`/`toId`, same `meta`).

#### `src/graph/GraphBuilder.ts`

**Implements:**

```typescript
export class GraphBuilder {
  constructor(private config: Config) {}

  async build(): Promise<{
    graph: Graph;
    chunks: Omit<ChunkRecord, 'embeddingText'>[];
  }>;
}
```

Logic:
1. Call `TsExtractor.extract(config)` → `{ nodes, rawChunks, project }`.
2. Call `EdgeExtractor.extractEdges(project, nodes)` → `tsEdges`.
3. Call `TestLinker.link(nodes, tsEdges)` → `testEdges`.
4. Build a `Graph`: add all nodes, then all `tsEdges`, then all `testEdges`.
5. Call `graph.computeHubDegrees()`.
6. Return `{ graph, chunks: rawChunks }`.

### Acceptance Checklist
- [ ] `GraphBuilder.build()` on a small TS project completes without error
- [ ] `graph.getDegree(id)` returns a non-zero value for any node that has at least one edge
- [ ] `graph.computeHubDegrees()` sets `node.hubDegree` correctly (a node with 10 edges has `hubDegree = 10`)
- [ ] Given a file `payment.test.ts` that imports from `payment.ts`, `TestLinker` emits at least one `TESTS` edge (Signal 2)
- [ ] Every `TESTS` edge has a corresponding `TESTED_BY` edge with `fromId`/`toId` swapped

---

## Phase 4 — Storage Layer (SQLite)

**Goal:** Persist all graph data and BM25 index entries to SQLite. Fast reads for graph traversal and retrieval.

### Files to Create

#### `src/storage/sqlite/schema.sql`

```sql
CREATE TABLE IF NOT EXISTS nodes (
  id             TEXT PRIMARY KEY,
  kind           TEXT NOT NULL,
  name           TEXT NOT NULL,
  qualified_name TEXT NOT NULL,
  signature      TEXT,
  jsdoc          TEXT,
  start_line     INTEGER NOT NULL,
  end_line       INTEGER NOT NULL,
  relative_path  TEXT NOT NULL,
  folder_hierarchy TEXT NOT NULL,  -- JSON array
  hub_degree     INTEGER DEFAULT 0,
  data_json      TEXT NOT NULL     -- full SymbolNode JSON for hydration
);

CREATE TABLE IF NOT EXISTS edges (
  id      TEXT PRIMARY KEY,
  from_id TEXT NOT NULL,
  to_id   TEXT NOT NULL,
  kind    TEXT NOT NULL,
  weight  REAL DEFAULT 1.0,
  meta_json TEXT,
  FOREIGN KEY (from_id) REFERENCES nodes(id),
  FOREIGN KEY (to_id)   REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS chunks (
  id             TEXT PRIMARY KEY,
  node_id        TEXT NOT NULL,
  chunk_index    INTEGER NOT NULL,
  total_chunks   INTEGER NOT NULL,
  text           TEXT NOT NULL,
  embedding_text TEXT NOT NULL,
  relative_path  TEXT NOT NULL,
  start_line     INTEGER NOT NULL,
  end_line       INTEGER NOT NULL,
  token_count    INTEGER NOT NULL,
  content_hash   TEXT NOT NULL,
  data_json      TEXT NOT NULL,
  FOREIGN KEY (node_id) REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS bm25_terms (
  term      TEXT PRIMARY KEY,
  idf       REAL NOT NULL,
  doc_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bm25_postings (
  term     TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  tf       REAL NOT NULL,
  PRIMARY KEY (term, chunk_id)
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_from   ON edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to     ON edges(to_id);
CREATE INDEX IF NOT EXISTS idx_chunks_node  ON chunks(node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_kind   ON nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_path   ON nodes(relative_path);
```

#### `src/storage/sqlite/migrations.ts`

```typescript
import Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { join } from 'path';

export function runMigrations(db: Database.Database): void {
  const schema = readFileSync(join(import.meta.dirname, 'schema.sql'), 'utf8');
  db.exec(schema);
  db.prepare(`INSERT OR IGNORE INTO meta VALUES ('schema_version', '1')`).run();
}
```

#### `src/storage/sqlite/SqliteStore.ts`

```typescript
import Database from 'better-sqlite3';
import type { SymbolNode, GraphEdge, ChunkRecord, BM25IndexEntry } from '../../types/index.js';
import { runMigrations } from './migrations.js';

export class SqliteStore {
  private db: Database.Database;

  constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    runMigrations(this.db);
  }

  // ─── Nodes ────────────────────────────────────────────────────────────────
  upsertNodes(nodes: SymbolNode[]): void;
  getNode(id: string): SymbolNode | undefined;
  getNodesByKind(kind: string): SymbolNode[];
  getNodesByPath(relativePath: string): SymbolNode[];
  getNodeCount(): number;

  // ─── Edges ────────────────────────────────────────────────────────────────
  upsertEdges(edges: GraphEdge[]): void;
  getOutEdges(fromId: string): GraphEdge[];
  getInEdges(toId: string): GraphEdge[];
  getEdgeCount(): number;

  // ─── Chunks ───────────────────────────────────────────────────────────────
  upsertChunks(chunks: ChunkRecord[]): void;
  getChunk(id: string): ChunkRecord | undefined;
  getChunksByNode(nodeId: string): ChunkRecord[];
  getChunksByIds(ids: string[]): ChunkRecord[];
  getChunkCount(): number;
  getAverageChunkTokenCount(): number;

  // ─── BM25 ─────────────────────────────────────────────────────────────────
  upsertBm25Index(entries: BM25IndexEntry[]): void;
  getBm25Term(term: string): BM25IndexEntry | undefined;
  getBm25TermsBatch(terms: string[]): BM25IndexEntry[];
  getCorpusSize(): number;
  getBm25TermCount(): number;

  // ─── Meta ─────────────────────────────────────────────────────────────────
  getMeta(key: string): string | undefined;
  setMeta(key: string, value: string): void;

  close(): void;
}
```

**Implementation notes:**
- All `upsert*` methods use `INSERT OR REPLACE`.
- Wrap multi-row inserts in `this.db.transaction(fn)()` for batch performance.
- `getChunksByIds` uses a single `SELECT ... WHERE id IN (${placeholders})`.
- `upsertBm25Index` deletes existing postings for affected terms before inserting new ones (within the same transaction).

### Acceptance Checklist
- [ ] `runMigrations` on a new in-memory database creates all tables without error
- [ ] `upsertNodes([n1, n2])` → `getNode(n1.id)` returns `n1` with correct fields
- [ ] `upsertEdges` + `getOutEdges` round-trip correctly
- [ ] Batch upsert of 1 000 chunks completes in < 200 ms (WAL + transaction)
- [ ] `getAverageChunkTokenCount()` returns a positive number after chunks are inserted

---

## Phase 5 — Embedding Pipeline & LanceDB

**Goal:** Build rich `embeddingText` for every chunk, embed with bounded concurrency, store vectors in LanceDB.

### Files to Create

#### `src/embedding/EmbeddingProvider.ts`

```typescript
export interface EmbeddingProvider {
  embed(texts: string[]): Promise<number[][]>;
  readonly dimensions: number;
  readonly modelName: string;
}
```

#### `src/embedding/OpenAiEmbeddingProvider.ts`

```typescript
import OpenAI from 'openai';
import type { EmbeddingProvider } from './EmbeddingProvider.js';

export class OpenAiEmbeddingProvider implements EmbeddingProvider {
  readonly dimensions: number;
  readonly modelName: string;
  private client: OpenAI;

  constructor(model = 'text-embedding-3-small', dimensions = 1536) {
    // Reads OPENAI_API_KEY from environment automatically.
    // If OPENAI_API_KEY is not set, throw: "OPENAI_API_KEY environment variable is not set."
    if (!process.env.OPENAI_API_KEY) throw new Error('OPENAI_API_KEY environment variable is not set.');
    this.client = new OpenAI();
    this.modelName = model;
    this.dimensions = dimensions;
  }

  async embed(texts: string[]): Promise<number[][]> {
    // Batch into groups of 100 (OpenAI API hard limit per request).
    // For each batch: client.embeddings.create({ model, input: batch }).
    // Return the data[i].embedding arrays in order.
  }
}
```

#### `src/embedding/EmbeddingTextBuilder.ts`

**Implements:** `build(chunk: Omit<ChunkRecord, 'embeddingText'>, node: SymbolNode): string`

Constructs a rich text string for embedding. Include ALL of the following:

```
[kind] [qualifiedName]
path: [relativePath]
folder: [folderHierarchy.join(' > ')]
[signature if present]
[jsDoc if present]

[chunk.text]
```

**Why include path in the embedding text:** Queries like "find the payment handler" will retrieve the right chunks even if the code itself never mentions "payment" — because the folder path `src/payments/handler.ts` is embedded alongside the code.

#### `src/embedding/EmbeddingPipeline.ts`

**Implements:**

```typescript
import pLimit from 'p-limit';

export class EmbeddingPipeline {
  constructor(
    private provider: EmbeddingProvider,
    private lanceStore: LanceStore,
    private sqliteStore: SqliteStore,
    private config: Config
  ) {}

  async run(
    rawChunks: Omit<ChunkRecord, 'embeddingText'>[],
    nodeMap: Map<string, SymbolNode>
  ): Promise<void>;
}
```

Logic:
1. Check `lanceStore.existsByHashes(rawChunks.map(c => c.contentHash))` to find which chunks already exist. Skip them (incremental update support).
2. For new chunks: call `EmbeddingTextBuilder.build(chunk, nodeMap.get(chunk.nodeId)!)` to produce `embeddingText`. Save the full `ChunkRecord` (with `embeddingText`) to SQLite via `sqliteStore.upsertChunks`.
3. Group new chunks into batches of 100. Use `pLimit(config.embeddingConcurrency)` to cap concurrent API calls.
4. For each batch: embed → build `LanceRecord[]` → `lanceStore.upsert(records)`.
5. After all batches, call `lanceStore.createIndex()`.

#### `src/storage/lancedb/LanceSchema.ts`

```typescript
import * as arrow from 'apache-arrow';

export function buildLanceSchema(dimensions: number): arrow.Schema {
  return new arrow.Schema([
    new arrow.Field('id',           new arrow.Utf8(),   false),
    new arrow.Field('node_id',      new arrow.Utf8(),   false),
    new arrow.Field('content_hash', new arrow.Utf8(),   false),
    new arrow.Field('relative_path',new arrow.Utf8(),   false),
    new arrow.Field('chunk_index',  new arrow.Int32(),  false),
    new arrow.Field('vector',
      new arrow.FixedSizeList(
        dimensions,
        new arrow.Field('item', new arrow.Float32(), false)
      ),
      false
    ),
  ]);
}
```

#### `src/storage/lancedb/LanceStore.ts`

```typescript
import * as lancedb from '@lancedb/lancedb';

export interface LanceRecord {
  id: string;
  node_id: string;
  content_hash: string;
  relative_path: string;
  chunk_index: number;
  vector: number[];
}

export interface LanceSearchResult {
  id: string;
  node_id: string;
  score: number;  // 1 - cosine_distance (higher = more similar)
}

export class LanceStore {
  private db!: lancedb.Connection;
  private table!: lancedb.Table;

  async open(dirPath: string, dimensions: number): Promise<void>;
    // lancedb.connect(dirPath) → open or create table named 'chunks'
    // If table doesn't exist, create with buildLanceSchema(dimensions)

  async upsert(records: LanceRecord[]): Promise<void>;
    // Delete existing rows matching ids, then table.add(records)

  async search(vector: number[], topK: number): Promise<LanceSearchResult[]>;
    // table.search(vector).limit(topK).toArray()
    // Map _distance → score = 1 - _distance

  async existsByHashes(hashes: string[]): Promise<Set<string>>;
    // SELECT content_hash FROM chunks WHERE content_hash IN (...)
    // Returns Set of hashes that already exist

  async createIndex(): Promise<void>;
    // table.createIndex('vector', { config: lancedb.IvfPq({ numPartitions: 256, numSubVectors: 96 }) })
    // Wrap in try-catch — silently skip if table has < 256 rows
}
```

### Acceptance Checklist
- [ ] `EmbeddingTextBuilder.build` produces a multi-line string that includes the `relativePath` and `chunk.text`
- [ ] `LanceStore.upsert` + `LanceStore.search` round-trip: insert 10 random vectors, search with one of them, get it back in top-1
- [ ] `EmbeddingPipeline.run` with a mocked provider (returns random vectors) upserts all chunks to LanceDB and saves `embeddingText` to SQLite
- [ ] Running the pipeline twice (incremental) skips already-embedded chunks (verify via mock call count = 0 on second run)

---

## Phase 6 — BM25 Inverted Index

**Goal:** Build a full inverted index over chunk text. Persist to SQLite. Expose a `Bm25Scorer` that ranks chunks given a query string.

### Files to Create

#### `src/search/bm25/Tokenizer.ts`

```typescript
const STOP_WORDS = new Set([
  'the','a','an','and','or','is','it','in','of','to','for',
  'this','that','with','as','at','be','by','on','from','are',
  'was','has','have','had','will','not','but','if','its',
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9_$]+/)        // split on anything that isn't an identifier char
    .filter(t => t.length > 1)    // drop single chars
    .filter(t => !STOP_WORDS.has(t));
}
```

#### `src/search/bm25/InvertedIndex.ts`

**Implements:** `class InvertedIndexBuilder { build(chunks: ChunkRecord[]): BM25IndexEntry[] }`

Logic:
1. For each chunk, call `tokenize(chunk.embeddingText)`.
2. Compute per-chunk term frequency: `tf(t, d) = count(t in d) / totalTerms(d)`.
3. Track document frequency: `df[t]` = number of chunks containing `t`.
4. After all chunks: `idf(t) = Math.log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)` where `N = chunks.length`.
5. Build and return `BM25IndexEntry[]` — one entry per unique term.

#### `src/search/bm25/Bm25Scorer.ts`

```typescript
export class Bm25Scorer {
  private readonly k1 = 1.2;
  private readonly b  = 0.75;
  private avgDocLen: number;

  constructor(private store: SqliteStore) {
    this.avgDocLen = store.getAverageChunkTokenCount();
  }

  score(queryText: string, topK: number): SearchHit[];
}
```

Logic:
1. Tokenize `queryText`.
2. Load BM25 entries for all unique query tokens via `store.getBm25TermsBatch(terms)`.
3. For each posting across all terms:
   ```
   score(t, d) += idf(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * docLen(d) / avgDocLen))
   ```
4. Accumulate scores per `chunkId`. Sort descending. Return top `topK`.
5. Hydrate each hit's `chunk` via `store.getChunksByIds`. Set `source: 'bm25'`.

### Acceptance Checklist
- [ ] `tokenize("export async function handlePayment()")` returns `['export', 'async', 'function', 'handlePayment']`
- [ ] Building an index from 3 chunks and scoring returns them in correct relevance order
- [ ] BM25 scores are always non-negative
- [ ] `Bm25Scorer.score('payment handler', 5)` returns hydrated `SearchHit[]` with `source: 'bm25'`

---

## Phase 7 — Hybrid Retrieval & Graph Expansion

**Goal:** Fuse BM25 and vector results. Expand the graph around seed nodes with a configurable budget. Assemble the final context string.

### Files to Create

#### `src/search/vector/VectorSearcher.ts`

```typescript
export class VectorSearcher {
  constructor(
    private lanceStore: LanceStore,
    private provider: EmbeddingProvider,
    private store: SqliteStore
  ) {}

  async search(queryText: string, topK: number): Promise<SearchHit[]>;
    // 1. provider.embed([queryText]) → vector
    // 2. lanceStore.search(vector, topK) → LanceSearchResult[]
    // 3. store.getChunksByIds(ids) → hydrate chunks
    // 4. Set source: 'vector'. Return SearchHit[].
}
```

#### `src/search/hybrid/RrfFusion.ts`

```typescript
export function rrfFusion(
  rankLists: SearchHit[][],  // one list per retriever; order = rank
  k: number,                 // RRF constant, default 60
  topK: number
): SearchHit[];
```

Logic (Reciprocal Rank Fusion):
1. For each hit at rank `r` (1-based) in each list: `rrf_score += 1 / (k + r)`.
2. Sum across all lists per unique `chunkId`.
3. Sort descending. Return top `topK` with `source: 'hybrid'`.

#### `src/expansion/BudgetConfig.ts`

```typescript
export interface ExpansionBudget {
  seedCap: number;
  hubDegreeCap: number;
  maxVisitedNodes: number;
  edgeKindWeights: Record<string, number>;
  excludedEdgeKinds: Set<string>;
}
```

#### `src/expansion/GraphExpander.ts`

**Implements:**

```typescript
export class GraphExpander {
  constructor(private store: SqliteStore, private budget: ExpansionBudget) {}
  expand(seedNodeIds: string[]): GraphContext;
}
```

Logic (priority-queue BFS with budget):
1. Cap seeds: `seeds = seedNodeIds.slice(0, budget.seedCap)`.
2. Initialize a max-heap (priority queue) sorted by edge weight. Seed it with all outgoing and incoming edges of seed nodes (loaded from SQLite).
3. Track `visited = new Set<string>(seeds)`.
4. While queue is non-empty and `visited.size < budget.maxVisitedNodes`:
   - Pop highest-priority edge.
   - Skip if `budget.excludedEdgeKinds.has(edge.kind)`.
   - Identify the neighbor node (the side not yet visited).
   - Load from SQLite: `store.getNode(neighborId)`.
   - Skip if `node.hubDegree > budget.hubDegreeCap`.
   - Skip if already visited.
   - Mark visited. Load neighbor's edges, multiply each by `budget.edgeKindWeights[edge.kind] ?? 1.0`, push to heap.
5. Return `GraphContext { seedNodeIds: seeds, visitedNodeIds: [...visited], edges: [traversed edges], nodes: [visited nodes] }`.

#### `src/search/hybrid/HybridRetriever.ts`

```typescript
export class HybridRetriever {
  constructor(
    private bm25: Bm25Scorer,
    private vector: VectorSearcher,
    private config: Config
  ) {}

  async retrieve(query: string): Promise<SearchHit[]> {
    const [bm25Hits, vectorHits] = await Promise.all([
      Promise.resolve(this.bm25.score(query, this.config.bm25TopK)),
      this.vector.search(query, this.config.vectorTopK),
    ]);
    return rrfFusion([bm25Hits, vectorHits], this.config.rrfK, this.config.hybridTopK);
  }
}
```

#### `src/query/ContextAssembler.ts`

**Implements:** `assemble(query: string, hits: SearchHit[], graphCtx: GraphContext): string`

Build the context string that will be passed to the LLM:

```
## Query
{query}

## Retrieved Code Chunks ({hits.length} results)

### Chunk {n+1}: {chunk.pathMeta.relativePath}:{chunk.startLine}–{chunk.endLine}
Node: {node.qualifiedName} ({node.kind})
Score: {hit.score.toFixed(4)} [{hit.source}]

```typescript
{chunk.text}
```

---

## Graph Context ({visitedNodeIds.length} nodes, {edges.length} edges)

{for each edge:}
- [{edge.kind}] {fromNode.qualifiedName} → {toNode.qualifiedName}

## Answer Instructions
Answer the query using only the code chunks and graph context above.
Cite specific locations as filename:line when relevant.
If the answer cannot be determined from the context, say so explicitly.
```

### Acceptance Checklist
- [ ] `VectorSearcher.search` returns `SearchHit[]` with `score` in `[0, 1]` and `source: 'vector'`
- [ ] `rrfFusion` with identical rank lists returns the same ordering
- [ ] `rrfFusion` with two conflicting lists promotes items appearing high in both
- [ ] `GraphExpander.expand` never visits more than `maxVisitedNodes` nodes
- [ ] `GraphExpander.expand` never returns a node with `hubDegree > hubDegreeCap`
- [ ] `GraphExpander.expand` omits edges in `excludedEdgeKinds`
- [ ] `ContextAssembler.assemble` output contains the query, at least one chunk text block, and the edge summary section

---

## Phase 8 — Query Engine, LLM Placeholder & CLI

**Goal:** Wire all components together. Expose a clean `QueryEngine.query()` API. Build a CLI with `index`, `query`, and `inspect` sub-commands. The `PlaceholderLlmProvider` returns the fully assembled context string without making any LLM call.

### Files to Create

#### `src/query/llm/LlmProvider.ts`

```typescript
export interface LlmProvider {
  complete(context: string, query: string): Promise<string>;
}
```

#### `src/query/llm/PlaceholderLlmProvider.ts`

```typescript
export class PlaceholderLlmProvider implements LlmProvider {
  async complete(context: string, _query: string): Promise<string> {
    // TODO: Replace with a real LLM call.
    // Returns the assembled context as-is so you can inspect exactly
    // what would be sent to the model.
    return [
      '=== PLACEHOLDER — NO LLM CONFIGURED ===',
      'Plug in a real LlmProvider to generate answers.',
      'Context that would be sent to the model:',
      '',
      context,
      '',
      '=== END PLACEHOLDER ===',
    ].join('\n');
  }
}
```

#### `src/query/QueryEngine.ts`

```typescript
export class QueryEngine {
  constructor(
    private retriever: HybridRetriever,
    private expander: GraphExpander,
    private assembler: ContextAssembler,
    private llm: LlmProvider,
    private config: Config
  ) {}

  async query(queryText: string): Promise<QueryResult> {
    // 1. Retrieve hybrid hits
    const hits = await this.retriever.retrieve(queryText);

    // 2. Collect seed node IDs (capped)
    const seedNodeIds = [...new Set(hits.map(h => h.nodeId))]
      .slice(0, this.config.expansionBudget.seedCap);

    // 3. Expand graph
    const graphContext = this.expander.expand(seedNodeIds);

    // 4. Assemble context string
    const assembledContext = this.assembler.assemble(queryText, hits, graphContext);

    // 5. LLM (placeholder or real)
    const llmAnswer = await this.llm.complete(assembledContext, queryText);

    return { query: queryText, hits, graphContext, assembledContext, llmAnswer };
  }
}
```

#### `src/indexer/Indexer.ts`

```typescript
export class Indexer {
  constructor(private config: Config) {}

  async run(): Promise<void> {
    // 1. GraphBuilder.build() → { graph, chunks }
    // 2. SqliteStore.upsertNodes(graph.allNodes())
    // 3. SqliteStore.upsertEdges(graph.allEdges())
    // 4. EmbeddingPipeline.run(chunks, nodeMap)
    //    → sets embeddingText on chunks, saves to SQLite, upserts to LanceDB
    // 5. Reload full ChunkRecord[] from SQLite (now have embeddingText)
    // 6. InvertedIndexBuilder.build(chunks) → BM25IndexEntry[]
    // 7. SqliteStore.upsertBm25Index(entries)
    // 8. SqliteStore.setMeta('last_indexed', new Date().toISOString())
    console.log('Index build complete.');
  }
}
```

#### `src/cli/index.ts`

```typescript
import { Command } from 'commander';

const program = new Command()
  .name('rag-pipeline')
  .description('TypeScript Codebase RAG Pipeline v3')
  .version('3.0.0');

// ─── index ───────────────────────────────────────────────────────────────────
program
  .command('index')
  .description('Build the full index from a TypeScript codebase')
  .requiredOption('-r, --root <path>', 'Path to codebase root')
  .option('-o, --output <path>', 'Output directory', '.rag-cache')
  .option('-c, --config <path>', 'Path to JSON config file')
  .action(async (opts) => {
    const config = loadConfig(opts);
    await new Indexer(config).run();
  });

// ─── query ───────────────────────────────────────────────────────────────────
program
  .command('query')
  .description('Query the index')
  .requiredOption('-q, --query <text>', 'Natural language query')
  .option('-o, --output <path>', 'Index directory', '.rag-cache')
  .option('--json', 'Output full result as JSON', false)
  .option('--top-k <n>', 'Number of results', '10')
  .action(async (opts) => {
    const config = loadConfig(opts);
    const engine = await buildQueryEngine(config);
    const result = await engine.query(opts.query);
    if (opts.json) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log(result.llmAnswer);
    }
  });

// ─── inspect ─────────────────────────────────────────────────────────────────
program
  .command('inspect')
  .description('Print index statistics')
  .option('-o, --output <path>', 'Index directory', '.rag-cache')
  .action(async (opts) => {
    const config = loadConfig(opts);
    const store = new SqliteStore(join(config.outputDir, 'metadata.db'));
    console.log('Nodes:       ', store.getNodeCount());
    console.log('Edges:       ', store.getEdgeCount());
    console.log('Chunks:      ', store.getChunkCount());
    console.log('BM25 terms:  ', store.getBm25TermCount());
    console.log('Last indexed:', store.getMeta('last_indexed') ?? 'never');
    store.close();
  });

program.parse();
```

#### `buildQueryEngine` factory (add to `cli/index.ts` or extract to `src/query/factory.ts`)

```typescript
async function buildQueryEngine(config: Config): Promise<QueryEngine> {
  const store      = new SqliteStore(join(config.outputDir, 'metadata.db'));
  const lanceStore = new LanceStore();
  await lanceStore.open(join(config.outputDir, 'vectors'), config.embeddingDimensions);

  const provider   = new OpenAiEmbeddingProvider(config.embeddingModel, config.embeddingDimensions);
  const vector     = new VectorSearcher(lanceStore, provider, store);
  const bm25       = new Bm25Scorer(store);
  const retriever  = new HybridRetriever(bm25, vector, config);

  const budget: ExpansionBudget = {
    seedCap:           config.expansionBudget.seedCap,
    hubDegreeCap:      config.expansionBudget.hubDegreeCap,
    maxVisitedNodes:   config.expansionBudget.maxVisitedNodes,
    edgeKindWeights:   config.expansionBudget.edgeKindWeights,
    excludedEdgeKinds: new Set(config.expansionBudget.excludedEdgeKinds),
  };
  const expander   = new GraphExpander(store, budget);
  const assembler  = new ContextAssembler();
  const llm        = new PlaceholderLlmProvider();   // ← swap this when ready

  return new QueryEngine(retriever, expander, assembler, llm, config);
}
```

### Acceptance Checklist
- [ ] `npx tsx src/cli/index.ts index --root ./my-ts-project` completes, creates `.rag-cache/metadata.db` and `.rag-cache/vectors/`
- [ ] `npx tsx src/cli/index.ts query --query "how does payment processing work"` prints the placeholder context block
- [ ] `npx tsx src/cli/index.ts query --query "..." --json` outputs valid JSON with `hits`, `graphContext`, `assembledContext`, `llmAnswer` keys
- [ ] `npx tsx src/cli/index.ts inspect` prints non-zero node/edge/chunk counts
- [ ] `llmAnswer` starts with `=== PLACEHOLDER — NO LLM CONFIGURED ===`
- [ ] `graphContext.visitedNodeIds.length` ≤ `config.expansionBudget.maxVisitedNodes`

---

## Integration Test: Full End-to-End

Create `tests/e2e/full-pipeline.test.ts`:

```typescript
import { describe, it, expect, beforeAll } from 'vitest';

describe('Full pipeline E2E', () => {
  const FIXTURE   = './tests/fixtures/sample-ts-project';
  const OUT_DIR   = './tests/.rag-cache-test';
  let engine: QueryEngine;

  beforeAll(async () => {
    const config = ConfigSchema.parse({ codebaseRoot: FIXTURE, outputDir: OUT_DIR });
    await new Indexer(config).run();
    engine = await buildQueryEngine(config);
  }, 120_000);

  it('returns hits for a known function name', async () => {
    const result = await engine.query('handlePayment');
    expect(result.hits.length).toBeGreaterThan(0);
    expect(result.hits[0].chunk.text).toContain('handlePayment');
  });

  it('respects graph expansion budget', async () => {
    const result = await engine.query('authentication');
    expect(result.graphContext.visitedNodeIds.length).toBeLessThanOrEqual(60);
  });

  it('assembled context contains path metadata', async () => {
    const result = await engine.query('authentication');
    expect(result.assembledContext).toContain(result.hits[0].chunk.pathMeta.relativePath);
  });

  it('placeholder LLM returns context string', async () => {
    const result = await engine.query('what does the search function do');
    expect(result.llmAnswer).toContain('PLACEHOLDER');
  });
});
```

---

## Plugging in a Real LLM (Post-Implementation)

Create `src/query/llm/OpenAiLlmProvider.ts`:

```typescript
import OpenAI from 'openai';
import type { LlmProvider } from './LlmProvider.js';

export class OpenAiLlmProvider implements LlmProvider {
  private client = new OpenAI();

  async complete(context: string, query: string): Promise<string> {
    const res = await this.client.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: 'You are a code assistant. Answer based only on the provided context.' },
        { role: 'user',   content: context },
      ],
      temperature: 0.1,
    });
    return res.choices[0].message.content ?? '';
  }
}
```

In `buildQueryEngine`, change one line:
```typescript
// Before:
const llm = new PlaceholderLlmProvider();
// After:
const llm = new OpenAiLlmProvider();
```

No other changes needed anywhere.

---

## Phase Summary

| Phase | What gets built | Key output |
|-------|----------------|------------|
| 1 | Scaffold, types, config | `Config`, `SymbolNode`, `GraphEdge`, `ChunkRecord` |
| 2 | ts-morph extraction | `SymbolNode[]`, `GraphEdge[]`, raw `ChunkRecord[]` |
| 3 | Graph + test linker | Unified `Graph`, hub degrees, `TESTS`/`TESTED_BY` edges |
| 4 | SQLite store | Persisted nodes / edges / chunks / BM25 tables |
| 5 | Embeddings + LanceDB | `embeddingText` on all chunks, vector index |
| 6 | BM25 inverted index | Tokenized inverted index, `Bm25Scorer` |
| 7 | Hybrid retrieval + graph expansion | RRF-fused hits, bounded `GraphContext` |
| 8 | QueryEngine + CLI | Working `index` / `query` / `inspect` commands, placeholder LLM |
