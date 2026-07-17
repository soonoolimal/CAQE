from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.base import ExperimentBase
from experiments.natural_stories_parser import NSSentence
from model import VQCAE


class RTAnalyzer(ExperimentBase):
    def __init__(self, backbone_key: str, backbone_cfg: dict, device: torch.device):
        super().__init__(backbone_key, backbone_cfg, device, dataset="natural_stories")
        self.fwd_bs = backbone_cfg["forward_batch_size"]

    @torch.no_grad()
    def prepare(self, sentences: list[NSSentence], cache_path: Path | None = None) -> dict:
        if cache_path is not None and cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"[{self.backbone_key}] Cache loaded: {cache_path} (n={len(cache['story_ids'])})")
            return cache

        self._load_llm_and_tokenizer()
        extract_fn = self._extract_mlm if self.backbone_type == "mlm" else self._extract_ntp

        story_ids, sent_ids, hidden_list, h_llm_list = [], [], [], []

        # iterate over 485 unique sentences once (not per-subject)
        pbar = tqdm(sentences, desc=f"[{self.backbone_key}]", dynamic_ncols=True)
        for sent in pbar:
            result = extract_fn(sent)
            if result is None or result[0].shape[0] == 0:
                tqdm.write(f"[{self.backbone_key}] Skipped ({sent.story_id}, {sent.sent_id})")
                continue
            h, h_llm = result
            story_ids.append(sent.story_id)
            sent_ids.append(sent.sent_id)
            hidden_list.append(h.cpu())
            h_llm_list.append(h_llm.cpu())

        cache = {
            "story_ids": story_ids,  # list[int]
            "sent_ids": sent_ids,  # list[int]
            "hidden": hidden_list,  # list[Tensor(n_i, h_dim)]
            "h_llm": h_llm_list,  # list[Tensor(n_i,)]
        }  # n_i: number of valid word positions

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache, cache_path)
            print(f"[{self.backbone_key}] Cache saved: {cache_path}")

        self.unload()
        return cache

    # per-sentence mean H_LLM: cached per-position entropy averaged within each sentence
    @staticmethod
    def compute_mean_llm_entropy(cache: dict) -> list[float]:
        return [h.mean().item() for h in cache["h_llm"]]

    # per-sentence mean CAQE: cached hidden -> compute_caqe per position -> averaged within each sentence
    @torch.no_grad()
    def compute_mean_caqe(self, cache: dict, vqcae: VQCAE) -> list[float]:
        vqcae.eval()
        return [vqcae.compute_caqe(h.to(self.device)).mean().item() for h in cache["hidden"]]

    # N_j masked forward passes -> (hidden, h_llm) of shape (n_valid, h_dim) and (n_valid,)
    def _extract_mlm(self, sent: NSSentence) -> tuple[torch.Tensor, torch.Tensor] | None:
        assert self.tokenizer is not None and self.llm is not None

        sentence_text = sent.text
        enc = self.tokenizer(
            sentence_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
        )
        input_ids = enc["input_ids"]  # (1, seq_len)
        attention_mask = enc["attention_mask"]  # (1, seq_len)
        offsets = enc["offset_mapping"][0].tolist()

        special_ids = set(self.tokenizer.all_special_ids)
        mask_token_id = self.tokenizer.mask_token_id

        # build one masked input per valid word position
        masked_ids_list, attn_mask_list, mask_positions = [], [], []
        for word_pos in range(sent.n_words):
            char_start, char_end = self._word_char_span(sentence_text, word_pos)
            token_pos = self._single_token_pos(offsets, char_start, char_end, special_ids, input_ids[0])
            # skip subword-split words (mirrors train/extractor.py)
            if token_pos is None:
                continue
            masked = input_ids[0].clone()
            masked[token_pos] = mask_token_id
            masked_ids_list.append(masked)
            attn_mask_list.append(attention_mask[0])
            mask_positions.append(token_pos)

        if not masked_ids_list:
            return None

        # batched forward pass: extract h at [MASK] and H_LLM per position
        all_h, all_entropy = [], []
        for start in range(0, len(masked_ids_list), self.fwd_bs):
            end = min(start + self.fwd_bs, len(masked_ids_list))
            batch_ids = torch.stack(masked_ids_list[start:end]).to(self.device)
            batch_mask = torch.stack(attn_mask_list[start:end]).to(self.device)
            batch_pos = mask_positions[start:end]

            out = self.llm(input_ids=batch_ids, attention_mask=batch_mask, output_hidden_states=True)

            rows = torch.arange(len(batch_pos), device=self.device)
            cols = torch.tensor(batch_pos, device=self.device)
            h_batch = out.hidden_states[-1][rows, cols].float().detach().cpu()
            all_h.append(h_batch)

            # compute H_LLM row-by-row (avoid materializing full (B, seq_len, vocab) softmax)
            for k, mpos in enumerate(batch_pos):
                log_p = F.log_softmax(out.logits[k, mpos].float(), dim=-1)
                all_entropy.append((-(log_p.exp() * log_p).sum()).detach().cpu())

        return torch.cat(all_h, dim=0), torch.stack(all_entropy)

    # 1 forward pass -> (hidden, h_llm) of shape (n_valid, h_dim) and (n_valid,)
    def _extract_ntp(self, sent: NSSentence) -> tuple[torch.Tensor, torch.Tensor] | None:
        assert self.tokenizer is not None and self.llm is not None

        sentence_text = sent.text
        enc = self.tokenizer(
            sentence_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()

        special_ids = set(self.tokenizer.all_special_ids)

        # single forward pass for the full sentence
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0].float()  # (seq_len, h_dim)

        # extract h_{t-1} and H_LLM per valid word position
        all_h, all_entropy = [], []
        for word_pos in range(sent.n_words):
            char_start, char_end = self._word_char_span(sentence_text, word_pos)
            token_pos = self._single_token_pos(offsets, char_start, char_end, special_ids, input_ids[0])
            # skip subword-split words and sentence-initial tokens (mirrors train/extractor.py)
            if token_pos is None or token_pos == 0:
                continue

            h = last_hidden[token_pos - 1].detach().cpu()  # h_{t-1}

            # compute H_LLM row-by-row (avoid materializing full (seq_len, vocab) softmax)
            log_p = F.log_softmax(out.logits[0, token_pos - 1].float(), dim=-1)
            entropy = (-(log_p.exp() * log_p).sum()).detach().cpu()

            all_h.append(h)
            all_entropy.append(entropy)

        if not all_h:
            return None
        return torch.stack(all_h), torch.stack(all_entropy)
