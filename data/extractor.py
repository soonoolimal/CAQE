from abc import ABC, abstractmethod
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class BaseHiddenExtractor(ABC):
    def __init__(self, model_key: str, data_cfg: dict):
        backbone_cfg = data_cfg["backbone"][model_key]
        self.model_key = model_key
        self.model_hf_id = backbone_cfg["hf_id"]
        self.tok_max_len = backbone_cfg["max_len"]
        self.tok_bs = backbone_cfg["tokenizer_batch_size"]
        self.fwd_bs = backbone_cfg["forward_batch_size"]

        save_cfg = data_cfg["save"]
        self.chunk_size = save_cfg["chunk_size"]
        self.save_dir = Path(save_cfg["dirs"]["hidden_vecs"]) / model_key

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None

    def _load_model_and_tokenizer(self):
        print(f"[{self.model_key}] loading model: {self.model_hf_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)

        # NTP models (GPT-family) have no pad_token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # AutoModel returns last_hidden_state without a prediction head
        # bfloat16 halves memory with minimal precision loss
        self.model = AutoModel.from_pretrained(self.model_hf_id, torch_dtype=torch.bfloat16)
        self.model.eval()
        self.model.to(self.device)
        print(f"[{self.model_key}] loaded -> {self.device}")

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.tokenizer is not None
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True, max_length=self.tok_max_len,
        )
        # token_type_ids omitted: each input is preprocessed as single sentence (not a sentence pair)
        # so it is all zeros, and NTP models don't even use it
        return encoded["input_ids"], encoded["attention_mask"]  # vocab indices of tokens, padding mask | (B, seq_len)

    def _save_hidden_chunk(self, data: dict, chunk_idx: int):
        save_path = self.save_dir / f"chunk_{chunk_idx:04d}.pt"
        torch.save(data, save_path)
        tqdm.write(f"[{self.model_key}] chunk saved: {save_path} | shape={tuple(data['hidden'].shape)}")

    def _save_metadata(self, h_dim: int, backbone_type: str):
        metadata = {
            "model_name": self.model_hf_id,
            "backbone_type": backbone_type,
            "h_dim": h_dim,
            "vocab_size": len(self.tokenizer),
        }
        save_path = self.save_dir / "metadata.yaml"
        with open(save_path, "w") as f:
            yaml.dump(metadata, f)
        print(f"[{self.model_key}] metadata saved: {save_path}")

    @abstractmethod
    def extract(self, texts: list[str]):
        pass


class MLMHiddenExtractor(BaseHiddenExtractor):
    """Hidden vector extractor for MLM backbones (BERT-family)."""
    def _make_masked_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_offset: int,
    ) -> dict:
        """
        For each content token in the batch,
        creates a masked copy of its sentence with that token replaced by [MASK].
        """
        special_ids = set(self.tokenizer.all_special_ids)
        mask_token_id = self.tokenizer.mask_token_id

        masked_input_ids = []        # input_ids with one token replaced by [MASK]
        masked_attention_masks = []  # corresponding attention masks
        target_token_ids = []        # original token id before masking (ground truth)
        sentence_ids = []            # global sentence index
        token_positions = []         # masked token position within the sequence

        batch_size, seq_len = input_ids.shape

        # i: sentence index in current batch (in [0, batch_size-1])
        # pos: position within the sequence (in [0, seq_len-1])
        for i in range(batch_size):
            for pos in range(seq_len):
                # skip non-content tokens
                if attention_mask[i, pos].item() == 0:  # skip padding token
                    continue
                token_id = input_ids[i, pos].item()
                if token_id in special_ids:             # skip special tokens
                    continue

                # mask
                masked_seq = input_ids[i].clone()
                masked_seq[pos] = mask_token_id

                # collect forward inputs
                masked_input_ids.append(masked_seq)
                masked_attention_masks.append(attention_mask[i])

                # collect metadata for this masked input
                target_token_ids.append(token_id)
                sentence_ids.append(sentence_offset + i)
                token_positions.append(pos)

        return {
            "masked_input_ids": masked_input_ids,
            "masked_attention_masks": masked_attention_masks,
            "target_token_ids": target_token_ids,
            "sentence_ids": sentence_ids,
            "token_positions": token_positions,
        }

    @torch.no_grad()
    def _forward_masked(self, masked_inputs: dict) -> torch.Tensor | None:
        """
        Extracts the hidden vector at each [MASK] position,
        the contextual representation inferred from surrounding context rather than the token embedding itself.
        """
        assert self.model is not None
        all_hidden = []

        masked_input_ids = masked_inputs["masked_input_ids"]
        masked_attention_masks = masked_inputs["masked_attention_masks"]
        token_positions = masked_inputs["token_positions"]

        if not masked_input_ids:
            tqdm.write(f"[{self.model_key}] warning: no content tokens found in batch, skipping")
            return None

        for start in range(0, len(masked_input_ids), self.fwd_bs):
            end = min(start + self.fwd_bs, len(masked_input_ids))
            batch_ids = torch.stack(masked_input_ids[start:end]).to(self.device)
            batch_mask = torch.stack(masked_attention_masks[start:end]).to(self.device)
            batch_positions = token_positions[start:end]

            outputs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
            last_hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            rows = torch.arange(len(batch_positions), device=self.device)  # masked input index in forward batch
            cols = torch.tensor(batch_positions, device=self.device)       # [MASK] position of each masked input
            all_hidden.append(last_hidden_states[rows, cols].detach().cpu())

        return torch.cat(all_hidden, dim=0)

    def extract(self, texts: list[str]):
        """Extracts hidden vectors for all content tokens in texts and saves them as chunked .pt files.

        Chunk .pt format:
            hidden           (N, hidden_dim)  hidden vector at the [MASK] position (contextual representation)
            target_token_ids (N)              original token id before masking
            sentence_ids     (N)              global sentence index
            token_positions  (N)              masked position within the sequence
            (N = total number of content tokens accumulated in this chunk)
        """
        if self.save_dir.exists():
            print(f"[{self.model_key}] output dir already exists, skipping: {self.save_dir}")
            return
        self.save_dir.mkdir(parents=True, exist_ok=False)

        self._load_model_and_tokenizer()

        buf_hidden = []
        buf_target = []
        buf_sent = []
        buf_pos = []
        chunk_idx = 0
        sentence_offset = 0

        pbar = tqdm(range(0, len(texts), self.tok_bs), desc=f"MLM [{self.model_key}]", dynamic_ncols=True)
        for batch_start in pbar:
            batch_texts = texts[batch_start:batch_start + self.tok_bs]
            input_ids, attention_mask = self._tokenize(batch_texts)
            masked_inputs = self._make_masked_inputs(input_ids, attention_mask, sentence_offset)
            hidden = self._forward_masked(masked_inputs)

            if hidden is not None:
                buf_hidden.append(hidden)
                buf_target.extend(masked_inputs["target_token_ids"])
                buf_sent.extend(masked_inputs["sentence_ids"])
                buf_pos.extend(masked_inputs["token_positions"])
            sentence_offset += len(batch_texts)

            total = sum(h.shape[0] for h in buf_hidden)
            if total >= self.chunk_size:
                self._flush_chunk(buf_hidden, buf_target, buf_sent, buf_pos, chunk_idx)
                chunk_idx += 1
                buf_hidden, buf_target, buf_sent, buf_pos = [], [], [], []

        if buf_hidden:
            self._flush_chunk(buf_hidden, buf_target, buf_sent, buf_pos, chunk_idx)

        self._save_metadata(h_dim=self.model.config.hidden_size, backbone_type="mlm")
        self.model = None
        torch.cuda.empty_cache()

    def _flush_chunk(self, buf_hidden, buf_target, buf_sent, buf_pos, chunk_idx):
        data = {
            "hidden": torch.cat(buf_hidden, dim=0).float(),
            "target_token_ids": torch.tensor(buf_target, dtype=torch.long),
            "sentence_ids": torch.tensor(buf_sent, dtype=torch.long),
            "token_positions": torch.tensor(buf_pos, dtype=torch.long),
            "model_name": self.model_hf_id,
            "backbone_type": "mlm",
        }
        self._save_hidden_chunk(data, chunk_idx)


class NTPHiddenExtractor(BaseHiddenExtractor):
    """Hidden vector extractor for NTP backbones (GPT-family)."""
    def _make_ntp_pairs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_offset: int,
    ) -> dict:
        """
        For each content token at position t in the batch,
        records the target token and the hidden extraction position t-1.
        """
        special_ids = set(self.tokenizer.all_special_ids)

        target_token_ids = []  # token id at position t (prediction target)
        sentence_ids = []      # global sentence index
        token_positions = []   # target token position t within the sequence
        hidden_positions = []  # hidden extraction position t-1
        batch_indices = []     # sentence index within the current tokenizer batch

        batch_size, seq_len = input_ids.shape

        # i: sentence index in current batch (in [0, batch_size-1])
        # pos: target position (starts at 1 since h_{t-1} for pos=0 does not exist)
        for i in range(batch_size):
            for pos in range(1, seq_len):
                # skip non-content tokens
                if attention_mask[i, pos].item() == 0:  # skip padding token
                    continue
                token_id = input_ids[i, pos].item()
                if token_id in special_ids:             # skip special tokens
                    continue

                # collect forward pass indices
                hidden_positions.append(pos - 1)  # col index into last_hidden_state
                batch_indices.append(i)           # sentence index within tokenizer batch (used to filter sel and compute rows)

                # collect metadata for this pair
                target_token_ids.append(token_id)
                sentence_ids.append(sentence_offset + i)
                token_positions.append(pos)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_token_ids": target_token_ids,
            "sentence_ids": sentence_ids,
            "token_positions": token_positions,
            "hidden_positions": hidden_positions,
            "batch_indices": batch_indices,
        }

    @torch.no_grad()
    def _forward_batch(self, pairs: dict) -> torch.Tensor | None:
        """
        Runs a forward pass on the original sequence and extracts the hidden vector at position t-1
        for each content token at position t.
        """
        assert self.model is not None
        all_hidden = []

        input_ids = pairs["input_ids"]
        attention_mask = pairs["attention_mask"]
        batch_indices = pairs["batch_indices"]
        hidden_positions = pairs["hidden_positions"]

        if not hidden_positions:
            tqdm.write(f"[{self.model_key}] warning: no content tokens found in batch, skipping")
            return None

        tok_bs = input_ids.shape[0]
        for start in range(0, tok_bs, self.fwd_bs):
            end = min(start + self.fwd_bs, tok_bs)
            sel = [k for k, bi in enumerate(batch_indices) if start <= bi < end]
            if not sel:
                continue

            batch_ids = input_ids[start:end].to(self.device)
            batch_mask = attention_mask[start:end].to(self.device)

            outputs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
            last_hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            rows = torch.tensor([batch_indices[k] - start for k in sel], device=self.device)  # sentence index within sub-batch
            cols = torch.tensor([hidden_positions[k] for k in sel], device=self.device)       # h_{t-1} position of each pair
            all_hidden.append(last_hidden_states[rows, cols].detach().cpu())

        return torch.cat(all_hidden, dim=0)

    def extract(self, texts: list[str]):
        """Extracts hidden vectors for all content tokens in texts and saves them as chunked .pt files.

        Chunk .pt format:
            hidden           (N, hidden_dim)  hidden vector at position t-1 (contextual representation)
            target_token_ids (N)              token id at position t (prediction target)
            sentence_ids     (N)              global sentence index
            token_positions  (N)              target token position t within the sequence
            hidden_positions (N)              hidden extraction position t-1
            (N = total number of content tokens accumulated in this chunk)
        """
        if self.save_dir.exists():
            print(f"[{self.model_key}] output dir already exists, skipping: {self.save_dir}")
            return
        self.save_dir.mkdir(parents=True, exist_ok=False)

        self._load_model_and_tokenizer()

        buf_hidden = []
        buf_target = []
        buf_sent = []
        buf_pos = []
        buf_hid_pos = []
        chunk_idx = 0
        sentence_offset = 0

        pbar = tqdm(range(0, len(texts), self.tok_bs), desc=f"NTP [{self.model_key}]", dynamic_ncols=True)
        for batch_start in pbar:
            batch_texts = texts[batch_start:batch_start + self.tok_bs]
            input_ids, attention_mask = self._tokenize(batch_texts)
            pairs = self._make_ntp_pairs(input_ids, attention_mask, sentence_offset)
            hidden = self._forward_batch(pairs)

            if hidden is not None:
                buf_hidden.append(hidden)
                buf_target.extend(pairs["target_token_ids"])
                buf_sent.extend(pairs["sentence_ids"])
                buf_pos.extend(pairs["token_positions"])
                buf_hid_pos.extend(pairs["hidden_positions"])
            sentence_offset += len(batch_texts)

            total = sum(h.shape[0] for h in buf_hidden)
            if total >= self.chunk_size:
                self._flush_chunk(buf_hidden, buf_target, buf_sent, buf_pos, buf_hid_pos, chunk_idx)
                chunk_idx += 1
                buf_hidden, buf_target, buf_sent, buf_pos, buf_hid_pos = [], [], [], [], []

        if buf_hidden:
            self._flush_chunk(buf_hidden, buf_target, buf_sent, buf_pos, buf_hid_pos, chunk_idx)

        self._save_metadata(h_dim=self.model.config.hidden_size, backbone_type="ntp")
        self.model = None
        torch.cuda.empty_cache()

    def _flush_chunk(self, buf_hidden, buf_target, buf_sent, buf_pos, buf_hid_pos, chunk_idx):
        data = {
            "hidden": torch.cat(buf_hidden, dim=0).float(),
            "target_token_ids": torch.tensor(buf_target, dtype=torch.long),
            "sentence_ids": torch.tensor(buf_sent, dtype=torch.long),
            "token_positions": torch.tensor(buf_pos, dtype=torch.long),
            "hidden_positions": torch.tensor(buf_hid_pos, dtype=torch.long),
            "model_name": self.model_hf_id,
            "backbone_type": "ntp",
        }
        self._save_hidden_chunk(data, chunk_idx)
