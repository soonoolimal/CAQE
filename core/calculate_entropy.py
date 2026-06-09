from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoModelForMaskedLM,
                          AutoTokenizer, PreTrainedTokenizerBase)

from data.extract_hidden import BaseHiddenExtractor
from models.caqe import CAQE


def shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


class BaseEntropyEstimator(ABC):
    """Base class for per-token Shannon entropy estimators."""

    @abstractmethod
    def run_from_texts(self, texts: list[str]) -> torch.Tensor:
        """Computes per-token Shannon entropy for all non-special tokens in texts.

        Returns:
            Entropy tensor of shape [N,].
        """


class FreqEntropy:
    """Computes corpus-level Shannon entropy from token frequency distribution.

    Tokenizes the corpus, counts unigram frequencies (excluding special tokens),
    and returns H of the resulting distribution.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self._counts: Counter = Counter()

    def fit(self, texts: list[str]) -> "FreqEntropy":
        """Tokenizes texts and accumulates token frequency counts.

        Can be called multiple times to accumulate over multiple batches.
        """
        special_ids = set(self.tokenizer.all_special_ids)
        encoded = self.tokenizer(texts, add_special_tokens=True)
        for ids in encoded["input_ids"]:
            self._counts.update(tid for tid in ids if tid not in special_ids)
        return self

    def entropy(self) -> float:
        """Returns Shannon entropy (nats) of the fitted frequency distribution."""
        assert self._counts, "Call fit() before entropy()."
        counts = torch.tensor(list(self._counts.values()), dtype=torch.float)
        probs = counts / counts.sum()
        return shannon_entropy(probs.unsqueeze(0)).item()


class MLMEntropy(BaseEntropyEstimator):
    """Computes per-token Shannon entropy from the MLM LM head output distribution.

    For each non-special token, masks it and runs a forward pass through the full
    MLM model (backbone + LM head). The entropy of the softmax distribution over
    the vocabulary at the [MASK] position is returned.
    """

    def __init__(self, model_hf_id: str, preprocess_config: dict):
        self.model_hf_id = model_hf_id
        self.tokenizer_max_length = preprocess_config["tokenizer_max_length"]
        self.tokenizer_batch_size = preprocess_config["tokenizer_batch_size"]
        self.forward_batch_size = preprocess_config["forward_batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_hf_id, dtype=torch.bfloat16)
        self.model.eval()
        self.model.to(self.device)

    def _make_masked_examples(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[list, list, list]:
        special_ids = set(self.tokenizer.all_special_ids)
        mask_token_id = self.tokenizer.mask_token_id
        masked_input_ids, masked_attn_masks, mask_positions = [], [], []
        for i in range(input_ids.shape[0]):
            for pos in range(input_ids.shape[1]):
                if attention_mask[i, pos].item() == 0:
                    continue
                if input_ids[i, pos].item() in special_ids:
                    continue
                masked_ids = input_ids[i].clone()
                masked_ids[pos] = mask_token_id
                masked_input_ids.append(masked_ids)
                masked_attn_masks.append(attention_mask[i].clone())
                mask_positions.append(pos)
        return masked_input_ids, masked_attn_masks, mask_positions

    def _forward_entropy(
        self, masked_input_ids: list, masked_attn_masks: list, mask_positions: list
    ) -> list[torch.Tensor]:
        all_entropy = []
        for start in range(0, len(masked_input_ids), self.forward_batch_size):
            end = start + self.forward_batch_size
            batch_ids = torch.stack(masked_input_ids[start:end]).to(self.device)
            batch_mask = torch.stack(masked_attn_masks[start:end]).to(self.device)
            positions = mask_positions[start:end]
            logits = self.model(input_ids=batch_ids, attention_mask=batch_mask).logits
            rows = torch.arange(len(positions), device=self.device)
            cols = torch.tensor(positions, device=self.device)
            probs = F.softmax(logits[rows, cols], dim=-1)  # [batch, vocab_size]
            all_entropy.append(shannon_entropy(probs).cpu())
        return all_entropy

    def run_from_texts(self, texts: list[str]) -> torch.Tensor:
        """Returns per-token entropy [N,] over all non-special tokens in texts."""
        if self.model is None:
            self._load_model()

        all_entropy = []
        with torch.no_grad():
            for batch_start in range(0, len(texts), self.tokenizer_batch_size):
                batch_texts = texts[batch_start: batch_start + self.tokenizer_batch_size]
                encoded = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.tokenizer_max_length,
                )
                masked_ids, masked_masks, positions = self._make_masked_examples(
                    encoded["input_ids"], encoded["attention_mask"],
                )
                all_entropy.extend(self._forward_entropy(masked_ids, masked_masks, positions))

        return torch.cat(all_entropy) if all_entropy else torch.empty(0)


class NTPEntropy(BaseEntropyEstimator):
    """Computes per-token Shannon entropy from the NTP LM head output distribution.

    For each non-special token t, runs a forward pass and takes the logits at
    position t-1. The entropy of the softmax distribution over the vocabulary
    at that position is returned as the entropy for token t.
    """

    def __init__(self, model_hf_id: str, preprocess_config: dict):
        self.model_hf_id = model_hf_id
        self.tokenizer_max_length = preprocess_config["tokenizer_max_length"]
        self.tokenizer_batch_size = preprocess_config["tokenizer_batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_hf_id, dtype=torch.bfloat16)
        self.model.eval()
        self.model.to(self.device)

    def _batch_entropy(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: set
    ) -> list[torch.Tensor]:
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        all_entropy = []
        for i in range(input_ids.shape[0]):
            for pos in range(1, input_ids.shape[1]):
                if attention_mask[i, pos].item() == 0:
                    continue
                if attention_mask[i, pos - 1].item() == 0:
                    continue
                if input_ids[i, pos].item() in special_ids:
                    continue
                all_entropy.append(shannon_entropy(probs[i, pos - 1].unsqueeze(0)).cpu())
        return all_entropy

    def run_from_texts(self, texts: list[str]) -> torch.Tensor:
        """Returns per-token entropy [N,] over all non-special tokens in texts."""
        if self.model is None:
            self._load_model()

        special_ids = set(self.tokenizer.all_special_ids)
        all_entropy = []
        with torch.no_grad():
            for batch_start in range(0, len(texts), self.tokenizer_batch_size):
                batch_texts = texts[batch_start: batch_start + self.tokenizer_batch_size]
                encoded = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.tokenizer_max_length,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                all_entropy.extend(self._batch_entropy(input_ids, attention_mask, special_ids))

        return torch.cat(all_entropy) if all_entropy else torch.empty(0)


class CAQEEntropy(BaseEntropyEstimator):
    """Computes CAQE entropy over pre-computed chunk files or raw text.

    Works for both MLM and NTP backbones — pass the corresponding extractor or chunk paths.
    """

    def __init__(self, model: CAQE, device: torch.device, extractor: BaseHiddenExtractor | None = None):
        self.model = model
        self.device = device
        self.extractor = extractor
        self.model.eval()
        self.model.to(device)

    def run(self, chunk_paths: list[Path]) -> torch.Tensor:
        """Computes CAQE entropy for all tokens across the given chunk files.

        Args:
            chunk_paths: List of .pt chunk files produced by MLMHiddenExtractor or NTPHiddenExtractor.

        Returns:
            Entropy tensor of shape [N,] over all tokens in the chunks.
        """
        entropies = []
        with torch.no_grad():
            for path in chunk_paths:
                data = torch.load(path, map_location="cpu", weights_only=True)
                h = data["hidden"].to(self.device)  # [N, hidden_dim]
                entropies.append(self.model.caqe(h).cpu())
        return torch.cat(entropies)

    def run_from_texts(self, texts: list[str]) -> torch.Tensor:
        """Extracts hidden vectors on-the-fly and computes CAQE entropy without saving to disk.

        Args:
            texts: Raw input sentences.

        Returns:
            Entropy tensor of shape [N,] over all non-special tokens.
        """
        assert self.extractor is not None, "Pass an extractor to CAQEEntropy to use run_from_texts()."
        with torch.no_grad():
            h = self.extractor.extract_to_memory(texts).to(self.device)  # [N, hidden_dim]
            return self.model.caqe(h).cpu()                               # [N]
