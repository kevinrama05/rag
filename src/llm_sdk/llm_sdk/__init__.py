from typing import Tuple, Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    logging,
)
from huggingface_hub import hf_hub_download

logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            if self._device in ['cuda', 'mps']:
                dtype = torch.float16
            else:
                dtype = torch.float32
        self._dtype = dtype

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> Any:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        input_tensor = torch.tensor(
            [input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]

    def get_path_to_vocab_file(self) -> Any:
        vocab_file_name = self._tokenizer.vocab_files_names.get(
            "vocab_file", "vocab.json"
        )
        vocab_path = hf_hub_download(
            repo_id=self._model_name, filename=vocab_file_name)
        return vocab_path

    def get_path_to_merges_file(self) -> Any:
        merges_file_name = self._tokenizer.vocab_files_names.get(
            "merges_file", "merges.txt"
        )
        merges_path = hf_hub_download(
            repo_id=self._model_name, filename=merges_file_name
        )
        return merges_path

    def get_path_to_tokenizer_file(self) -> Any:
        tokenizer_file_name = self._tokenizer.vocab_files_names.get(
            "tokenizer_file", "tokenizer.json"
        )
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name, filename=tokenizer_file_name
        )
        return tokenizer_path

    def get_logits_with_cache(
        self,
        input_ids: list[int],
        past_key_values: Optional[Any] = None,
    ) -> Tuple[list[float], Any]:
        """Compute next-token logits using an optional KV cache.

        On the first call, pass the full prompt as `input_ids` with
        `past_key_values=None`. On every subsequent call, pass only the
        single most-recently generated token id, along with the
        `past_key_values` returned by the previous call — this avoids
        reprocessing the whole growing sequence at each decoding step.

        Args:
            input_ids: Token ids to feed this step (full prompt on the
                first call, a single new token id afterward).
            past_key_values: Cache from a previous call, or None to start
                a fresh sequence.

        Returns:
            A tuple of (next-token logits, updated past_key_values cache)
            to be passed back into the next call.
        """
        input_tensor = torch.tensor(
            [input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits], out.past_key_values
