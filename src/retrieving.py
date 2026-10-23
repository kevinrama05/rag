import json
from .pydantic_validations import MinimalSource
import bm25s
from pydantic import BaseModel


class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    first_character_index: int
    last_character_index: int
    score: float
    text: str


class Retriever:
    def __init__(
        self,
        index_path: str = "data/processed",
    ) -> None:
        self.retriever = bm25s.BM25.load(
            index_path,
            load_corpus=True,
        )

    def load_and_search(
        self,
        index_dir: str,
        query: str, k: int = 5, print_results: bool = True
    ) -> list[MinimalSource]:
        retriever = bm25s.BM25.load(index_dir, load_corpus=False)

        with open(f"{index_dir}/sources.json", "r", encoding="utf-8") as f:
            raw_sources = json.load(f)
        sources = [MinimalSource(**s) for s in raw_sources]

        query_tokens = bm25s.tokenize(query)
        doc_indices, scores = retriever.retrieve(query_tokens, k=k)

        sources_ = []
        for i in range(doc_indices.shape[1]):
            idx = doc_indices[0, i]
            source = sources[idx]
            sources_.append(source)
            first_idx = source.first_character_index
            last_idx = source.last_character_index
            if print_results:
                print(
                    f"{source.file_path} [{first_idx}:{last_idx}]"
                )

        return sources_
