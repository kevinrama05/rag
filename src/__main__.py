import fire
from . import main_utils

if __name__ == "__main__":
    fire.Fire(
        {
            "index": main_utils.index,
            "search": main_utils.retrieve,
            "answer": main_utils.answer,
            "search_dataset": main_utils.search_dataset,
            "answer_dataset": main_utils.answer_dataset,
        }
    )
