from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.base import ExperimentBase
from experiments.coinco_parser import _SURROUNDING_PUNCT, CoinCoToken, find_target_position
from model import VQCAE


class Evaluator(ExperimentBase):
    """Compute per-token signed delta_p = P(target_token) - P(top1_substitute_token) over CoinCo entries."""

    def __init__(self, backbone_key: str, backbone_cfg: dict, device: torch.device):
        super().__init__(backbone_key, backbone_cfg, device, dataset="coinco")
        # a single LM-head model yields both the LM baseline logits and the backbone hidden state
        # (hidden_states[-1] is bit-identical to AutoModel.last_hidden_state, the h the VQCAE was trained on)
        self._load_llm_and_tokenizer()
        self._cache: dict | None = None  # populated by prepare()

    @torch.no_grad()
    def prepare(self, entries: list[CoinCoToken], cache_path: Path | None = None) -> None:
        """Run backbone forward passes for all valid CoinCo entries, cache hidden states and LLM baseline probabilities."""
        if cache_path is not None and cache_path.exists():
            self._cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"[{self.backbone_key}] Cache loaded: {cache_path} (n={len(self._cache['target_words'])})")
            return

        llm = self.llm
        assert llm is not None
        target_words, substitute_words = [], []
        hidden, hidden_sub, original_ids, substitute_ids = [], [], [], []
        llm_deltas, llm_p_targets, llm_p_subs = [], [], []

        pbar = tqdm(entries, desc=f"[{self.backbone_key}] Prepare", dynamic_ncols=True)
        for entry in pbar:
            # skip if target position not found
            word_pos = find_target_position(entry.sentence, entry.target_word, entry.pos_tag)
            if word_pos is None:
                continue

            # skip multi-word substitutes
            substitute_word, _ = entry.substitutes[0]
            if " " in substitute_word:
                continue

            # tokenize target sentence
            if self.backbone_type == "mlm":
                prep = self._tokenize_mlm(entry.sentence, word_pos)
            else:
                prep = self._tokenize_ntp(entry.sentence, word_pos)
            if prep is None:
                continue

            input_ids, attention_mask, pos, original_token_id = prep

            # tokenize parallel sentence (substitute replaces target at word_pos)
            sub_sentence = self._build_substitute_sentence(entry.sentence, word_pos, substitute_word)
            if self.backbone_type == "mlm":
                prep_sub = self._tokenize_mlm(sub_sentence, word_pos)
            else:
                prep_sub = self._tokenize_ntp(sub_sentence, word_pos)
            if prep_sub is None:
                continue

            # substitute_token_id from in-sentence tokenization (4th element of prep_sub)
            input_ids_sub, attention_mask_sub, pos_sub, substitute_token_id = prep_sub

            # backbone forward pass for target: extract h and LLM baseline prob
            out = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            h = out.hidden_states[-1][0, pos].float()
            probs = F.softmax(out.logits[0, pos].float(), dim=-1)

            # backbone forward pass for substitute: extract h_sub and LLM baseline prob
            out_sub = llm(input_ids=input_ids_sub, attention_mask=attention_mask_sub, output_hidden_states=True)
            h_sub = out_sub.hidden_states[-1][0, pos_sub].float()
            probs_sub = F.softmax(out_sub.logits[0, pos_sub].float(), dim=-1)

            target_words.append(entry.target_word)
            substitute_words.append(substitute_word)
            hidden.append(h.cpu())
            hidden_sub.append(h_sub.cpu())
            original_ids.append(original_token_id)
            substitute_ids.append(substitute_token_id)

            p_t = probs[original_token_id].item()
            p_s = probs_sub[substitute_token_id].item()
            llm_deltas.append(p_t - p_s)
            llm_p_targets.append(p_t)
            llm_p_subs.append(p_s)

        self._cache = {
            "target_words": target_words,
            "substitute_words": substitute_words,
            "hidden": torch.stack(hidden),  # (N, h_dim) target
            "hidden_sub": torch.stack(hidden_sub),  # (N, h_dim) substitute
            "original_ids": torch.tensor(original_ids),
            "substitute_ids": torch.tensor(substitute_ids),
            "llm_deltas": llm_deltas,
            "llm_p_targets": llm_p_targets,
            "llm_p_subs": llm_p_subs,
        }

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._cache, cache_path)
            print(f"[{self.backbone_key}] Cache saved: {cache_path} (n={len(target_words)})")

    @torch.no_grad()
    def evaluate(self, vqcae: VQCAE, batch_size: int = 512) -> list[tuple[str, str, float, float, float]]:
        """Return (target_word, substitute_word, delta_p, p_target, p_sub) for each cached token, using the VQCAE head."""
        assert self._cache is not None, "call prepare() before evaluate()"
        vqcae.eval()
        c = self._cache
        original_ids = c["original_ids"].to(self.device)
        substitute_ids = c["substitute_ids"].to(self.device)

        deltas, p_targets, p_subs = [], [], []
        for start in range(0, len(c["hidden"]), batch_size):
            h = c["hidden"][start : start + batch_size].to(self.device)
            h_sub = c["hidden_sub"][start : start + batch_size].to(self.device)

            z_e = vqcae.encode(h)
            z_q, _, _, _ = vqcae.quantize(z_e)
            probs_orig = F.softmax(vqcae.classifier(vqcae.decode(z_q)), dim=-1)

            z_e_sub = vqcae.encode(h_sub)
            z_q_sub, _, _, _ = vqcae.quantize(z_e_sub)
            probs_sub = F.softmax(vqcae.classifier(vqcae.decode(z_q_sub)), dim=-1)

            rows = torch.arange(h.shape[0], device=self.device)
            p_orig = probs_orig[rows, original_ids[start : start + batch_size]]
            p_sub = probs_sub[rows, substitute_ids[start : start + batch_size]]
            deltas.extend((p_orig - p_sub).cpu().tolist())
            p_targets.extend(p_orig.cpu().tolist())
            p_subs.extend(p_sub.cpu().tolist())

        return list(zip(c["target_words"], c["substitute_words"], deltas, p_targets, p_subs))

    def evaluate_llm(self) -> list[tuple[str, str, float, float, float]]:
        """Return (target_word, substitute_word, delta_p, p_target, p_sub) for each cached token, using the LM-head baseline."""
        assert self._cache is not None, "call prepare() before evaluate_llm()"
        c = self._cache
        return list(zip(c["target_words"], c["substitute_words"], c["llm_deltas"], c["llm_p_targets"], c["llm_p_subs"]))

    def _tokenize_mlm(self, sentence: str, word_pos: int) -> tuple[torch.Tensor, torch.Tensor, int, int] | None:
        """Return (masked_input_ids, attention_mask, mask_pos, original_token_id), or None if subword-split."""
        assert self.tokenizer is not None

        enc = self.tokenizer(
            sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=self.max_len
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # word -> token mapping via offset_mapping (evaluator-specific)
        char_start, char_end = self._word_char_span(sentence, word_pos)
        token_pos = self._single_token_pos(
            enc["offset_mapping"][0].tolist(),
            char_start,
            char_end,
            set(self.tokenizer.all_special_ids),
            input_ids[0],
        )
        if token_pos is None:
            return None
        original_token_id = input_ids[0, token_pos].item()

        # mirrors extractor: clone input_ids, mask at token_pos
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, token_pos] = self.tokenizer.mask_token_id

        return masked_input_ids.to(self.device), attention_mask.to(self.device), token_pos, original_token_id

    def _tokenize_ntp(self, sentence: str, word_pos: int) -> tuple[torch.Tensor, torch.Tensor, int, int] | None:
        """Return (input_ids, attention_mask, hidden_pos, original_token_id), or None if subword-split or first token."""
        assert self.tokenizer is not None

        enc = self.tokenizer(
            sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=self.max_len
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # word -> token mapping via offset_mapping (evaluator-specific)
        char_start, char_end = self._word_char_span(sentence, word_pos)
        target_token_pos = self._single_token_pos(
            enc["offset_mapping"][0].tolist(),
            char_start,
            char_end,
            set(self.tokenizer.all_special_ids),
            input_ids[0],
        )
        if target_token_pos is None or target_token_pos == 0:
            return None

        # mirrors extractor: token_id = input_ids[pos], hidden_pos = pos - 1
        original_token_id = input_ids[0, target_token_pos].item()
        hidden_pos = target_token_pos - 1

        return input_ids, attention_mask, hidden_pos, original_token_id

    def _build_substitute_sentence(self, sentence: str, word_pos: int, substitute_word: str) -> str:
        """Replace the split()-indexed target word while preserving surrounding punctuation."""
        words = sentence.split()
        original = words[word_pos]

        core = original.strip(_SURROUNDING_PUNCT)
        left_len = len(original) - len(original.lstrip(_SURROUNDING_PUNCT))
        right_len = len(original) - len(original.rstrip(_SURROUNDING_PUNCT))

        if not core:
            replacement = substitute_word
        else:
            prefix = original[:left_len]
            suffix = original[len(original) - right_len :] if right_len else ""
            replacement = f"{prefix}{substitute_word}{suffix}"

        return " ".join(words[:word_pos] + [replacement] + words[word_pos + 1 :])
