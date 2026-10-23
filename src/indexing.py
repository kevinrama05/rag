import bm25s
import os
from tqdm import tqdm
import ast
import json
from .pydantic_validations import MinimalSource
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class BM25:
    EXCLUDED_DIRS = {
        ".venv",
        ".git",
        "__pycache__",
        ".mypy_cache",
        "node_modules",
        ".pytest_cache",
        "build",
        "dist",
        "*.egg-info",
    }

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.filepaths: list[str] = []
        self.max_chunk_size = max_chunk_size
        self.sources: list[MinimalSource] = []
        self.corpus: list[str] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            add_start_index=True,
            chunk_size=max_chunk_size,
            chunk_overlap=min(400, max_chunk_size),
            length_function=len,
            is_separator_regex=False,
        )

    def _list_files(self) -> None:
        root_directory = "data/raw/vllm-0.10.1"

        print("Searching:", os.path.abspath(root_directory))

        if not os.path.isdir(root_directory):
            exit(
                "Directory does not exist: "
                f"{os.path.abspath(root_directory)}"
            )

        for root, dirs, files in os.walk("data/raw/vllm-0.10.1"):
            dirs[:] = [
                directory
                for directory in dirs
                if directory not in self.EXCLUDED_DIRS
                and not directory.endswith(".egg-info")
            ]

            for filename in files:
                if filename.endswith(".py") or filename.endswith(".md"):
                    filepath = os.path.join(root, filename)
                    self.filepaths.append(filepath)

    def _chunk_readme_file(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        documents: list[Document] = self.text_splitter.create_documents(
            [content])

        for document in documents:
            start = document.metadata["start_index"]
            end = start + len(document.page_content)

            self.corpus.append(document.page_content)

            self.sources.append(
                MinimalSource(
                    file_path=filepath,
                    first_character_index=start,
                    last_character_index=end,
                )
            )

    def _chunk_python_file(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as file:
            source = file.read()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            self._chunk_readme_file(filepath)
            return

        lines = source.splitlines(keepends=True)
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        def node_span(node: ast.AST) -> tuple[int, int]:
            start = line_offsets[node.lineno - 1] + node.col_offset
            end_lineno = node.end_lineno if node.end_lineno else node.lineno
            end_col = node.end_col_offset if node.end_col_offset else 0
            end = line_offsets[end_lineno - 1] + end_col
            return start, end

        def add_chunk(start: int, end: int, header: str | None = None) -> None:
            """Emit chunks covering [start, end), each capped at max_chunk_size.
            `header` is prepended to the INDEXED text only -- the stored
            MinimalSource offsets always point at raw file content, never the
            header, so scoring stays exact against the corpus."""
            while start < end:
                chunk_end = min(start + self.max_chunk_size, end)
                raw_text = source[start:chunk_end]
                indexed_text = f"{header}\n{raw_text}" if header else raw_text

                self.sources.append(
                    MinimalSource(
                        file_path=filepath,
                        first_character_index=start,
                        last_character_index=chunk_end,
                    )
                )
                self.corpus.append(indexed_text)
                start = chunk_end

        is_def = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

        # --- Pass 1: full top-level coverage — guarantees every character in
        # the file is indexed exactly once, with no chunk over max_chunk_size.
        last_end = 0
        top_level_ids: set[int] = set()

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, is_def):
                continue
            top_level_ids.add(id(node))
            start, end = node_span(node)

            if start > last_end:
                gap_text = source[last_end:start]
                if gap_text.strip():
                    add_chunk(last_end, start)

            header = f"File: {filepath}\nClass/Function: {node.name}"
            add_chunk(start, end, header=header)
            last_end = end

        if last_end < len(source):
            trailing = source[last_end:]
            if trailing.strip():
                add_chunk(last_end, len(source))

        # --- Pass 2: extra fine-grained chunks for NESTED defs (methods inside
        # classes, closures, etc.) -- improves precision when a question targets
        # a specific inner function/method by name.
        for node in ast.walk(tree):
            if not isinstance(node, is_def) or id(node) in top_level_ids:
                continue
            start, end = node_span(node)
            header = f"File: {filepath}\nClass/Function: {node.name}"
            add_chunk(start, end, header=header)

    def index_files(self) -> None:
        self._list_files()

        for file in tqdm(self.filepaths, desc="Chunking files", unit="files"):
            if file.endswith(".md"):
                self._chunk_readme_file(file)
            if file.endswith(".py"):
                self._chunk_python_file(file)

        if not self.corpus:
            raise RuntimeError("Corpus is empty!")
        corpus_tokens = bm25s.tokenize(self.corpus)
        retriever = bm25s.BM25(corpus=self.corpus)
        retriever.index(corpus_tokens)
        retriever.save("data/processed")
        sources_path = "data/processed/sources.json"
        with open(sources_path, "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in self.sources], f)
