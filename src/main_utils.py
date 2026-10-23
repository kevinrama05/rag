import json
from .indexing import BM25
from .retrieving import Retriever
from sys import exit
from .llm_sdk.llm_sdk import Small_LLM_Model
from .pydantic_validations import (
    MinimalAnswer,
    MinimalSource,
    MinimalSearchResults,
    AnsweredQuestion,
    UnansweredQuestion,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
import uuid
from pydantic import ValidationError
from tqdm import tqdm
import os
import gc
import torch


def prompt_generator(
    query: str,
    sources: list[MinimalSource],
) -> str:
    """Create a RAG prompt without exceeding the token budget."""

    system_prompt = (
        "You are a precise assistant answering questions about the vLLM "
        "codebase using ONLY the source excerpts provided below.\n"
        "Rules:\n"
        "- Base your answer strictly on the given excerpts. Do not invent "
        "APIs, files, or behavior that is not shown.\n"
        "- If the excerpts do not contain enough information to answer, "
        "say so explicitly instead of guessing.\n"
        "- After each factual claim, cite the source it came from in the "
        "form (Source N), matching the [Source N] labels below.\n"
        "- Be concise and direct. Prefer short explanations over long "
        "explanations.\n- Keep the answer under 2000 characters."
    )

    prompt_suffix = "<|im_start|>assistant\n" "<think>\n\n</think>\n\n"

    sources_text = ""

    for i in sources:
        with open(i.file_path, 'r') as file:
            content = file.read()
        first_idx = i.first_character_index
        last_idx = i.last_character_index
        sources_text += content[first_idx:last_idx] + '\n'

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>system\n{sources_text}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"{prompt_suffix}"
    )


# Program utils
def index(max_chunk_size: int = 2000) -> None:
    if max_chunk_size <= 0 or max_chunk_size > 2000:
        exit("--max_chunk_size should be between 1 and 2000")
    bm25 = BM25(max_chunk_size)
    bm25.index_files()
    print("Ingestion complete! Indices saved under data/processed")


def retrieve(
    query: str, k: int = 10, return_val: bool = False, print_val: bool = True
) -> list[MinimalSource] | None:
    retriever = Retriever()
    sources = retriever.load_and_search(
        "data/processed", query, k, print_results=print_val
    )
    if return_val:
        return sources
    else:
        return None


def answer(query: str, k: int = 10) -> str:
    try:
        sdk = Small_LLM_Model()

        sources = retrieve(query, k, return_val=True, print_val=False)
        if not sources:
            return ""
        minimal_search_result = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query, retrieved_sources=sources
        )
        prompt_ids = sdk.encode(prompt_generator(query, sources))[0].tolist()
        answer_ids: list[int] = []
        stop_id = sdk.encode("<|im_end|>")[0].item()

        past_key_values = None
        next_input = prompt_ids

        for _ in range(2000):
            logits, past_key_values = sdk.get_logits_with_cache(
                next_input, past_key_values
            )

            next_token_id = max(range(len(logits)), key=lambda i: logits[i])

            if next_token_id == stop_id:
                break

            answer_ids.append(next_token_id)
            next_input = [next_token_id]
        answer = sdk.decode(answer_ids)
        validate_answer = MinimalAnswer(
            question_id=minimal_search_result.question_id,
            question=minimal_search_result.question,
            retrieved_sources=minimal_search_result.retrieved_sources,
            answer=answer,
        )
        return validate_answer.answer

    except Exception as exc:
        print(f"answer failed for query {query!r}: {exc}")
        return ""


def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    try:
        with open(dataset_path, "r") as file:
            questions = json.load(file)

        rag_questions: list[AnsweredQuestion | UnansweredQuestion] = []
        for i in tqdm(
            questions["rag_questions"],
            desc="Validating questions", unit="question"
        ):
            try:
                rag_questions.append(UnansweredQuestion.model_validate(i))
            except ValidationError:
                continue

        rag_questions_valid = RagDataset(rag_questions=rag_questions)

        questions_search_results: list[MinimalSearchResults] = []
        for j in tqdm(
            rag_questions_valid.rag_questions,
            desc="Retrieving sources",
            unit="question",
        ):
            retrieved_sources = retrieve(
                j.question, k, return_val=True, print_val=False
            )
            questions_search_results.append(
                MinimalSearchResults(
                    question_id=j.question_id,
                    question=j.question,
                    retrieved_sources=retrieved_sources,
                )
            )

        validated_search_result = StudentSearchResults(
            search_results=questions_search_results, k=k
        )

        os.makedirs(save_directory, exist_ok=True)
        save_file = os.path.join(
            save_directory, os.path.basename(dataset_path))

        with open(save_file, "w") as file:
            json.dump(validated_search_result.model_dump(), file, indent=4)

        print(f"Saved student_search_results to {save_file}")

    except Exception as exc:
        print(f"search_dataset failed: {exc}")


def answer_dataset_helper(
    minimal_search: MinimalSearchResults, sdk: Small_LLM_Model
) -> MinimalAnswer:
    past_key_values = None
    answer_text = ""

    try:
        prompt = prompt_generator(
            minimal_search.question,
            minimal_search.retrieved_sources
        )

        prompt_ids = sdk.encode(prompt)[0].tolist()
        answer_ids: list[int] = []
        stop_id = sdk.encode("<|im_end|>")[0].item()

        next_input = prompt_ids

        for _ in range(2000):
            logits, past_key_values = sdk.get_logits_with_cache(
                next_input, past_key_values
            )

            next_token_id = max(range(len(logits)), key=lambda i: logits[i])

            if next_token_id == stop_id:
                break

            answer_ids.append(next_token_id)
            next_input = [next_token_id]

        answer_text = sdk.decode(answer_ids)

    except Exception as exc:
        print(f"\n[ERROR] Failed to answer question "
              f"{minimal_search.question_id}")
        print(f"  Exception: {type(exc).__name__}: {exc}")
        print(f"  Prompt tokens: {len(prompt_ids)}")
        print(f"  Retrieved sources: {len(minimal_search.retrieved_sources)}")

        past_key_values = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        answer_text = ""

    finally:
        past_key_values = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return MinimalAnswer(
        question=minimal_search.question,
        question_id=minimal_search.question_id,
        retrieved_sources=minimal_search.retrieved_sources,
        answer=answer_text,
    )


def answer_dataset(
        student_search_results_path: str, save_directory: str) -> None:
    try:
        with open(student_search_results_path, "r") as file:
            raw_search_results = json.load(file)

        student_search_results = StudentSearchResults.model_validate(
            raw_search_results)

        sdk = Small_LLM_Model()

        question_answered: list[MinimalAnswer] = []
        for search_result in tqdm(
            student_search_results.search_results,
            desc="Answering questions",
            unit="question",
        ):
            question_answered.append(answer_dataset_helper(search_result, sdk))
        validated_search_and_answer_results = StudentSearchResultsAndAnswer(
            search_results=question_answered, k=student_search_results.k
        )

        os.makedirs(save_directory, exist_ok=True)
        save_file = os.path.join(
            save_directory, os.path.basename(student_search_results_path)
        )

        with open(save_file, "w") as file:
            json.dump(
                validated_search_and_answer_results.model_dump(),
                file, indent=4)
        print(
            f"Loaded {len(student_search_results.search_results)} questions"
            f" ... Processed {len(question_answered)} of "
            f"{len(student_search_results.search_results)} questions"
        )
        print(f"Saved student_search_results_and_answers to {save_file}")
    except Exception as exc:
        print(f"answer_dataset failed: {exc}")
