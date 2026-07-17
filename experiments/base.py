from abc import ABC, abstractmethod
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from model import VQCAE

_ROOT = Path(__file__).parents[1]


class ExperimentBase(ABC):
    def __init__(self, backbone_key: str, backbone_cfg: dict, device: torch.device, dataset: str):
        self.backbone_key = backbone_key
        self.backbone_type = backbone_cfg["type"]
        self.hf_id = backbone_cfg["hf_id"]
        self.max_len = backbone_cfg["max_len"]

        self.device = device
        self.tokenizer = None
        self.llm = None

        self._set_dirs(dataset)

    def _set_dirs(self, dataset: str):
        self.raw_dir = _ROOT / "data" / "experiments" / "raw" / dataset
        self.h_dir = _ROOT / "data" / "experiments" / "hidden_vecs" / dataset

    def _load_llm_and_tokenizer(self):
        print(f"[{self.backbone_key}] Loading {self.hf_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        head_cls = AutoModelForMaskedLM if self.backbone_type == "mlm" else AutoModelForCausalLM
        self.llm = head_cls.from_pretrained(self.hf_id, dtype=torch.bfloat16)
        self.llm.eval()
        self.llm.to(self.device)

        print(f"[{self.backbone_key}] Loaded -> {self.device}")

    def unload(self):
        self.llm = None
        self.tokenizer = None
        torch.cuda.empty_cache()

    # run backbone forward passes and cache results
    @abstractmethod
    def prepare(self, *args, **kwargs) -> dict | None: ...

    @staticmethod
    def load_model(ckpt_path: Path, model: VQCAE) -> VQCAE:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model.eval()
        return model

    # return token position if the word maps to exactly one non-special token, else None
    def _single_token_pos(
        self,
        offsets: list[tuple[int, int]],
        char_start: int,
        char_end: int,
        special_ids: set[int],
        input_ids_row: torch.Tensor,
    ) -> int | None:
        # collect token candidates overlapping the word's char span
        positions = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start and e > s]
        if len(positions) != 1:
            return None

        # reject subword splits and special tokens
        token_pos = positions[0]
        if input_ids_row[token_pos].item() in special_ids:
            return None
        return token_pos

    # return (char_start, char_end) of the word at word_pos in sentence_text
    def _word_char_span(self, sentence_text: str, word_pos: int) -> tuple[int, int]:
        words = sentence_text.split()
        pos = 0
        for i, word in enumerate(words):
            idx = sentence_text.index(word, pos)
            if i == word_pos:
                return idx, idx + len(word)
            pos = idx + len(word)
        raise ValueError(f"word_pos {word_pos} out of range for: {sentence_text!r}")
