*This activity has been created as part of the 42 curriculum by kerama.*

# RAG against the machine

## Description

**RAG against the machine** is a Retrieval-Augmented Generation (RAG) system that answers
natural-language questions about the [vLLM](https://github.com/vllm-project/vllm) codebase.

A language model only "knows" what it saw during training, and retraining it every time a
codebase changes is impractical. Instead of baking knowledge into the model, this project
lets a small local model (`Qwen/Qwen3-0.6B`) *reach out* for the right information at answer
time. The pipeline has four stages:

1. **Indexing** — walk the vLLM repository, split every `.py` and `.md` file into chunks, and
   persist a searchable BM25 index.
2. **Retrieving** — given a question, return the top-k most relevant source locations
   (`file_path` + character span).
3. **Augmenting** — assemble the retrieved snippets into a token-budgeted prompt.
4. **Generating** — run `Qwen/Qwen3-0.6B` over that prompt to produce a grounded,
   structured JSON answer.

Retrieval quality is measured with **recall@k**, comparing retrieved source spans against a
ground-truth dataset via the provided moulinette.

## System Architecture

```
data/raw/vllm-0.10.1/  (corpus)
        │
        ▼
   src/indexing.py  ── BM25 class ─────────────► data/processed/ (bm25s index + sources.json)
        │
        ▼
   src/retrieving.py ── Retriever class ───────► ranked MinimalSource results
        │
        ▼
   src/main_utils.py ── prompt_generator() ────► token-budgeted prompt
        │
        ▼
   src/llm_sdk/llm_sdk ── Small_LLM_Model ─────► Qwen/Qwen3-0.6B (manual greedy decode + KV cache)
        │
        ▼
   Structured JSON answer (pydantic-validated)
```

- **`src/indexing.py`** — `BM25` class: walks `data/raw/vllm-0.10.1`, chunks every file, tokenizes
  the corpus with `bm25s`, and persists the index plus a `sources.json` sidecar (chunk metadata:
  `file_path`, `first_character_index`, `last_character_index`).
- **`src/retrieving.py`** — `Retriever` class: loads the persisted index and `sources.json`,
  tokenizes the query, and maps BM25's top-k document indices back to `MinimalSource` entries.
- **`src/main_utils.py`** — orchestration layer and the functions exposed via the CLI
  (`index`, `retrieve`/`search`, `answer`, `search_dataset`, `answer_dataset`), plus
  `prompt_generator`, which reads the retrieved spans back off disk and assembles the
  chat-formatted prompt for the model.
- **`src/llm_sdk/llm_sdk/__init__.py`** — `Small_LLM_Model`: thin wrapper around
  `transformers.AutoModelForCausalLM` / `AutoTokenizer` for `Qwen/Qwen3-0.6B`, exposing
  `encode`/`decode` and a KV-cache-aware `get_logits_with_cache` used to drive manual greedy
  decoding.
- **`src/pydantic_validations.py`** — the shared data contracts (`MinimalSource`,
  `UnansweredQuestion`/`AnsweredQuestion`, `RagDataset`, `StudentSearchResults`,
  `StudentSearchResultsAndAnswer`, etc.) that every stage validates against, matching the
  schema the moulinette expects.
- **`src/__main__.py`** — wires the functions above to a `python-fire` CLI.

## Chunking Strategy

Two distinct chunking strategies are implemented, matched to the file type:

- **Markdown / text chunking** (`BM25._chunk_readme_file`): uses LangChain's
  `RecursiveCharacterTextSplitter` with `chunk_size=max_chunk_size` and
  `chunk_overlap=min(400, max_chunk_size)`. `add_start_index=True` lets us recover the exact
  character offset of each chunk in the source file, which becomes
  `first_character_index`/`last_character_index` on the resulting `MinimalSource`.

- **Python code chunking** (`BM25._chunk_python_file`): parses the file with `ast.parse` and
  walks top-level `FunctionDef` / `AsyncFunctionDef` / `ClassDef` nodes. Each definition becomes
  its own chunk boundary (using line/column offsets converted to character offsets), and any gap
  between definitions (imports, module-level statements) is captured as its own chunk so nothing
  is dropped. Any chunk — whether a definition or a gap — that would exceed `max_chunk_size` is
  further split into fixed-size windows by `add_chunk`, guaranteeing no chunk ever exceeds the
  configured limit. Files that fail to parse (syntax errors) fall back to the Markdown/text
  chunker instead of being skipped.

`max_chunk_size` is a configurable CLI argument (default `2000`, matching the moulinette's
`max_context_length`) and is validated on `index` — values outside `1–2000` are rejected before
indexing starts, since chunks above 2000 characters would make the moulinette reject the entire
output.

## Retrieval Method

Retrieval uses **BM25** (via the [`bm25s`](https://github.com/xhluca/bm25s) library), one of the
two lexical methods allowed by the subject.

- At index time, the full corpus of chunk texts is tokenized with `bm25s.tokenize` and indexed
  with `bm25s.BM25(corpus=...).index(...)`, then persisted under `data/processed/`. Because
  `bm25s` only persists tokenized corpus text (not arbitrary metadata), chunk provenance
  (`file_path`, character spans) is stored separately in `data/processed/sources.json`, keyed by
  the same ordering as the indexed corpus.
- At search time, the query is tokenized the same way, `retriever.retrieve(query_tokens, k=k)`
  returns the top-k document indices and BM25 scores, and those indices are mapped back to their
  corresponding `MinimalSource` entries from `sources.json` for output.

## Performance Analysis

*(Fill in with your actual numbers from `./moulinette evaluate_student_search_results`, as
shown in the subject's example output. Do not commit fabricated figures.)*

| Metric | Docs questions | Code questions | Threshold |
|---|---|---|---|
| Recall@1  | `TODO` | `TODO` | — |
| Recall@3  | `TODO` | `TODO` | — |
| Recall@5  | `TODO` | `TODO` | ≥ 80% (docs) / ≥ 50% (code) |
| Recall@10 | `TODO` | `TODO` | — |

- **Indexing time**: `TODO` (must stay under 5 minutes for the whole corpus).
- **Retrieval throughput**: `TODO` (must stay under 90 seconds for 200 questions).
- Effect of `--max_chunk_size` on recall: `TODO` — describe how smaller chunk sizes affected
  recall@k in your own testing (smaller chunks are allowed, but the subject notes their effect
  on recall should be reported here).

## Design Decisions

- **Two-file separation of retrieval concerns**: `BM25` (`indexing.py`) is only responsible for
  building and persisting the index; `Retriever` (`retrieving.py`) is only responsible for
  loading an index and answering queries against it. `main_utils.py` orchestrates both and owns
  all CLI-facing I/O, keeping indexing/retrieval logic reusable and independently testable.
- **`sources.json` sidecar file**: `bm25s` persists only tokenized corpus text, not custom
  metadata, so chunk provenance (file path + character span) is tracked in a parallel JSON file
  indexed in lockstep with the corpus, and re-hydrated into `MinimalSource` objects at query
  time.
- **Manual greedy decoding with KV-cache reuse** (`get_logits_with_cache`) instead of
  `model.generate()`: this gives explicit control over stopping on the `<|im_end|>` token and
  avoids reprocessing the whole growing sequence at every decoding step, which matters given
  `Qwen/Qwen3-0.6B`'s limited reasoning budget and the project's throughput requirement.
  Reasoning/thinking content is skipped via an explicit empty `<think></think>` block in the
  prompt suffix to keep answers concise.
- **Explicit citation instructions in the system prompt**: the model is instructed to cite
  `(Source N)` after each factual claim and to say so explicitly rather than guess when the
  retrieved context is insufficient — aimed directly at the subject's "grounded, no major
  hallucination" answer criteria.
- **Pydantic models as the single source of truth for I/O**: every stage (`search`, `answer`,
  `search_dataset`, `answer_dataset`) reads and writes through the shared models in
  `pydantic_validations.py`, so output shape stays consistent with what the moulinette expects
  without duplicating validation logic per command.

## Challenges Faced

- **Chunk-size guarantees across AST boundaries**: a single function or class can be far larger
  than `max_chunk_size` on its own, so AST-boundary chunking alone isn't sufficient — a
  secondary fixed-size windowing pass (`add_chunk`) was layered on top of AST boundaries to
  guarantee every chunk written to the index respects the configured (and moulinette-enforced)
  size limit, regardless of how large the underlying code construct is.
- **Exact character-offset bookkeeping**: since the moulinette compares `file_path` verbatim and
  scores overlap by character span (IoU ≥ 0.05), offsets have to survive multiple layers of
  splitting (AST node boundaries, gap-filling between nodes, and secondary windowing) without
  drifting — an off-by-one here silently breaks recall scoring rather than raising an error.
- **Getting a small, reasoning-limited model to answer faithfully**: `Qwen/Qwen3-0.6B` has known
  reasoning limits, so prompt design (explicit "don't invent APIs," explicit citation format,
  explicit length cap) mattered more than sampling strategy for keeping answers grounded and on
  point.
- **Degenerate-input handling**: query/dataset commands need to survive empty queries, malformed
  JSON, and missing files without an unhandled traceback, which pushed most of `main_utils.py`'s
  top-level functions into `try/except` wrappers with per-question error isolation in the batch
  paths.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) as the package/project manager
- The vLLM repository extracted under `data/raw/vllm-0.10.1/`

### Install

```bash
make install
# or directly:
uv sync
```

### Index the corpus

```bash
uv run python -m src index --max_chunk_size 2000
```

Builds the BM25 index and `sources.json` sidecar under `data/processed/`.

### Search a single query

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

### Search a whole dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### Answer a single query

```bash
uv run python -m src answer "How to configure the OpenAI server?" --k 5
```

### Generate answers for a dataset

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### Score with the moulinette

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

### Makefile shortcuts

```bash
make install      # uv sync
make run           # run the CLI
make debug         # run under pdb
make lint          # flake8 + mypy (standard flags)
make clean         # remove __pycache__, .mypy_cache, etc.
```

> **Known limitation**: the `evaluate` CLI command described in the subject (for local
> recall@k iteration against a ground-truth dataset) is not yet exposed in `src/__main__.py`.
> Retrieval quality is currently checked only via the external `moulinette` executable, per the
> command above.

## Resources

- [vLLM documentation](https://docs.vllm.ai/) — the indexed corpus.
- [BM25S: Orders of magnitude faster lexical search via eager sparse scoring](https://arxiv.org/abs/2407.03618)
  — the BM25 implementation used (`bm25s`).
- [Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
  — background on the BM25 ranking function.
- [LangChain `RecursiveCharacterTextSplitter` docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
  — used for Markdown/text chunking.
- [Python `ast` module documentation](https://docs.python.org/3/library/ast.html) — used for
  structure-aware Python code chunking.
- [Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401)
  — the original RAG paper.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the mandatory generation model.
- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers) —
  `AutoModelForCausalLM`/`AutoTokenizer` usage and KV-cache (`past_key_values`) generation.

**How AI was used**: AI assistance (Claude) was used to debug `mypy`/`flake8` errors (e.g.
resolving a `return-value` error from an implicit `None` return, and diagnosing an
`UnboundLocalError` risk in an exception handler referencing a variable that might not yet be
assigned), to review error-handling paths in `search_dataset`/`answer_dataset` for
degenerate-input robustness, and to discuss chunking-strategy and Makefile design trade-offs.
All AI-suggested changes were reviewed, tested, and understood before being incorporated; no
whole functions were copy-pasted without verification.