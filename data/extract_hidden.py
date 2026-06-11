"""
Hidden vector extractors for MLM(BERT-family) and NTP(GPT-family) backbones.
Extracts token-level hidden vectors and saves them as chunked .pt files.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class BaseHiddenExtractor(ABC):
    """Base class for hidden vector extractors (via MLM and NTP).

    Handles model loading, tokenization, and chunk saving as shared logic.
    """
    def __init__(self, model_key: str, model_hf_id: str, preprocess_config: dict):
        self.model_key = model_key      # backbone (e.g., "bert")
        self.model_hf_id = model_hf_id  # HuggingFace model path (e.g., "bert-base-uncased")

        self.tokenizer_max_length = preprocess_config["tokenizer_max_length"]  # max sequence length for tokenizer
        self.tokenizer_batch_size = preprocess_config["tokenizer_batch_size"]  # sentences per tokenizer call
        self.forward_batch_size = preprocess_config["forward_batch_size"]      # forward batch size for MLM masked inputs
        self.chunk_size = preprocess_config["chunk_size"]                      # samples per .pt chunk file

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None  # loaded lazily on extract()
        self.model = None

    def _load_model(self):
        print(f"Loading model: {self.model_hf_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)

        # Fall back to eos_token since NTP models have no pad_token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # AutoModel: base transformer without LM head, returns last_hidden_state.
        # bf16 halves memory and doubles Tensor Core throughput with hidden vectors cast to float32 on save
        self.model = AutoModel.from_pretrained(self.model_hf_id, dtype=torch.bfloat16)
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded: {self.model_hf_id} -> {self.device}")

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.tokenizer is not None, "_load_model() must be called before _tokenize()."
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,     # pad to the longest sequence in the batch
            truncation=True,  # truncate sequences exceeding max_length
            max_length=self.tokenizer_max_length,
        )
        # input_ids[i, j]: vocab token id at position j in sentence i
        # attention_mask[i, j]: 1 for real tokens, 0 for padding
        return encoded["input_ids"], encoded["attention_mask"]

    # full hidden vector extraction pipeline
    @abstractmethod
    def extract(self, texts: list[str], out_dir: Path):
        pass

    def _save_chunk(self, data: dict, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        hidden_shape = tuple(data["hidden"].shape)
        tqdm.write(f"Chunk saved: {out_path} | hidden shape={hidden_shape}")


class MLMHiddenExtractor(BaseHiddenExtractor):
    """Hidden vector extractor for MLM backbones (BERT-family).

    For each non-special token, replaces it with [MASK] and runs a forward pass.
    The hidden vector at the [MASK] position is produced by the frozen pretrained LLM.

    In MLM models, h_[MASK] serves as the token's contextual representation,
    with the original (pre-mask) token as the classification target.

    Chunk .pt format:
        hidden           : [N, hidden_dim] ─ hidden vector at the mask position
        target_token_ids : [N]             — original token id before masking
        sentence_ids     : [N]             — global sentence index
        token_positions  : [N]             — masked position (index within sequence)
        model_name       : str
        backbone_type    : "mlm"
    """
    # generates masked input variants and metadata for each non-special token in the batch
    def _make_examples(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_offset: int,
    ) -> tuple[list, list, list, list, list]:
        # set of special token ids to skip (CLS, SEP, PAD, MASK, etc.)
        special_ids = set(self.tokenizer.all_special_ids)
        mask_token_id = self.tokenizer.mask_token_id

        masked_input_ids = []   # input_ids with one token replaced by [MASK]
        masked_attn_masks = []  # corresponding attention masks
        target_token_ids = []   # original token id before masking (ground truth)
        sentence_ids = []       # global sentence index
        token_positions = []    # position of the masked token within the sequence

        batch_size, seq_len = input_ids.shape

        for i in range(batch_size):
            for pos in range(seq_len):
                token_id = input_ids[i, pos].item()

                # skip padding
                if attention_mask[i, pos].item() == 0:
                    continue

                # skip special tokens (CLS, SEP, etc.)
                if token_id in special_ids:
                    continue

                # clone the original sequence and replace one token with [MASK]
                masked_ids = input_ids[i].clone()
                masked_ids[pos] = mask_token_id

                masked_input_ids.append(masked_ids)
                masked_attn_masks.append(attention_mask[i].clone())
                target_token_ids.append(token_id)
                sentence_ids.append(sentence_offset + i)
                token_positions.append(pos)

        return masked_input_ids, masked_attn_masks, target_token_ids, sentence_ids, token_positions

    # runs forward passes on masked inputs in sub-batches and extracts hidden at each [MASK] position
    @torch.no_grad()
    def _forward_masked(
        self,
        masked_input_ids: list[torch.Tensor],
        masked_attn_masks: list[torch.Tensor],
        token_positions: list[int],
    ) -> torch.Tensor:
        assert self.model is not None, "_load_model() must be called before _forward_masked()."
        all_hidden = []

        # run forward in forward_batch_size sub-batches
        for start in range(0, len(masked_input_ids), self.forward_batch_size):
            end = start + self.forward_batch_size

            # stack list of tensors into [forward_batch_size, seq_len] and move to device
            batch_ids = torch.stack(masked_input_ids[start:end]).to(self.device)
            batch_mask = torch.stack(masked_attn_masks[start:end]).to(self.device)
            batch_positions = token_positions[start:end]

            # extract hidden vectors from the frozen pretrained LLM
            outputs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
            last_hidden = outputs.last_hidden_state  # [forward_batch_size, seq_len, hidden_dim]

            # gather hidden vectors at mask positions with a single vectorized index op
            rows = torch.arange(len(batch_positions), device=self.device)
            cols = torch.tensor(batch_positions, device=self.device)
            all_hidden.append(last_hidden[rows, cols].detach().cpu())

        if not all_hidden:
            return torch.empty(0)

        return torch.cat(all_hidden, dim=0)  # [N, hidden_dim], N = total non-special tokens in batch

    # runs the full extraction pipeline: tokenize -> mask -> forward -> accumulate -> save chunks
    def extract(self, texts: list[str], out_dir: Path) -> None:
        out_dir = out_dir / self.model_key
        if out_dir.exists():
            print(f"[skip] {out_dir} already exists")
            return
        out_dir.mkdir(parents=True, exist_ok=False)

        self._load_model()

        # accumulation buffers for chunk saving
        buf_hidden = []
        buf_target = []
        buf_sent = []
        buf_pos = []

        chunk_idx = 0        # index of the next chunk file to save
        sentence_offset = 0  # global sentence index offset across batches

        for batch_start in tqdm(range(0, len(texts), self.tokenizer_batch_size), desc=f"MLM [{self.model_key}]", dynamic_ncols=True):
            batch_texts = texts[batch_start: batch_start + self.tokenizer_batch_size]

            # tokenize batch
            input_ids, attention_mask = self._tokenize(batch_texts)

            # generate masked examples
            masked_ids, masked_masks, target_ids, sent_ids, tok_positions = self._make_examples(
                input_ids, attention_mask, sentence_offset,
            )

            # forward masked inputs and extract hidden vectors at mask positions
            if masked_ids:
                hidden = self._forward_masked(masked_ids, masked_masks, tok_positions)

                buf_hidden.append(hidden)
                buf_target.extend(target_ids)
                buf_sent.extend(sent_ids)
                buf_pos.extend(tok_positions)

            sentence_offset += len(batch_texts)

            # flush to disk when accumulated samples reach chunk_size
            total = sum(h.shape[0] for h in buf_hidden)
            if total >= self.chunk_size:
                self._flush_chunk(buf_hidden, buf_target, buf_sent, buf_pos, out_dir, chunk_idx)
                chunk_idx += 1
                buf_hidden, buf_target, buf_sent, buf_pos = [], [], [], []

        # save any remaining samples as the final chunk
        if buf_hidden:
            self._flush_chunk(buf_hidden, buf_target, buf_sent, buf_pos, out_dir, chunk_idx)

        del self.model
        self.model = None
        torch.cuda.empty_cache()

    # concatenates buffers and saves one chunk .pt file
    def _flush_chunk(self, buf_hidden, buf_target, buf_sent, buf_pos, out_dir, chunk_idx):
        hidden = torch.cat(buf_hidden, dim=0)   # [N, hidden_dim]
        data = {
            "hidden": hidden.float(),
            "target_token_ids": torch.tensor(buf_target, dtype=torch.long),
            "sentence_ids": torch.tensor(buf_sent, dtype=torch.long),
            "token_positions": torch.tensor(buf_pos, dtype=torch.long),
            "model_name": self.model_hf_id,
            "backbone_type": "mlm",
        }
        out_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
        self._save_chunk(data, out_path)

    @torch.no_grad()
    def extract_to_memory(self, texts: list[str]) -> torch.Tensor:
        """Extracts MLM hidden vectors into memory without saving to disk.

        Returns:
            hidden: [N, hidden_dim] float tensor over all non-special tokens.
        """
        if self.model is None:
            self._load_model()

        all_hidden = []
        for batch_start in range(0, len(texts), self.tokenizer_batch_size):
            batch_texts = texts[batch_start: batch_start + self.tokenizer_batch_size]
            input_ids, attention_mask = self._tokenize(batch_texts)
            masked_ids, masked_masks, _, _, tok_positions = self._make_examples(
                input_ids, attention_mask, sentence_offset=0,
            )
            if masked_ids:
                hidden = self._forward_masked(masked_ids, masked_masks, tok_positions)
                all_hidden.append(hidden.float())

        return torch.cat(all_hidden, dim=0) if all_hidden else torch.empty(0)


class NTPHiddenExtractor(BaseHiddenExtractor):
    """Hidden vector extractor for NTP backbones (GPT-family).

    For each token t, runs a forward pass on the original sequence without masking.
    The hidden vector at position t-1 is produced by the frozen pretrained LLM.

    In NTP models, h_{t-1} serves as the token's contextual representation,
    with token t as the classification target.

    Chunk .pt format:
        hidden           : [N, hidden_dim] — hidden vector at position t-1
        target_token_ids : [N]             — token id at position t (prediction target)
        sentence_ids     : [N]             — global sentence index
        token_positions  : [N]             — target token position (t)
        hidden_positions : [N]             — hidden extraction position (t-1)
        model_name       : str
        backbone_type    : "ntp"
    """
    # generates (target, h_{t-1}) pairs and metadata for each non-special token in the batch
    def _make_examples(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_offset: int,
    ) -> tuple[list, list, list, list, list]:
        # set of special token ids to skip (BOS, EOS, PAD, etc.)
        special_ids = set(self.tokenizer.all_special_ids)

        target_token_ids = []  # token id at position t (prediction target)
        sentence_ids = []      # global sentence index
        token_positions = []   # target position t
        hidden_positions = []  # hidden extraction position t-1
        batch_indices = []     # local batch index for indexing into last_hidden_state

        batch_size, seq_len = input_ids.shape

        for i in range(batch_size):
            for pos in range(1, seq_len):  # skip pos=0: no h_{-1} exists
                token_id = input_ids[i, pos].item()

                # skip padding at target position
                if attention_mask[i, pos].item() == 0:
                    continue

                # skip if preceding position is also padding
                if attention_mask[i, pos - 1].item() == 0:
                    continue

                # skip special tokens
                if token_id in special_ids:
                    continue

                target_token_ids.append(token_id)
                sentence_ids.append(sentence_offset + i)
                token_positions.append(pos)       # target position t
                hidden_positions.append(pos - 1)  # hidden position t-1
                batch_indices.append(i)           # for indexing last_hidden_state[i, pos-1]

        return target_token_ids, sentence_ids, token_positions, hidden_positions, batch_indices

    # runs a single forward pass on the full batch and extracts hidden at each h_{t-1} position
    @torch.no_grad()
    def _forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_indices: list[int],
        hidden_positions: list[int],
    ) -> torch.Tensor:
        assert self.model is not None, "_load_model() must be called before _forward_batch()."

        if not hidden_positions:
            return torch.empty(0)

        # extract hidden vectors from the frozen pretrained LLM
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [tokenizer_batch_size, seq_len, hidden_dim]

        # index h_{t-1} for each sample: last_hidden[batch_indices[k], hidden_positions[k]]
        bi = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
        hp = torch.tensor(hidden_positions, dtype=torch.long, device=self.device)
        hidden = last_hidden[bi, hp]  # [N, hidden_dim], N = total non-special tokens in batch

        return hidden.detach().cpu()

    # runs the full extraction pipeline: tokenize -> forward -> accumulate -> save chunks
    def extract(self, texts: list[str], out_dir: Path):
        out_dir = out_dir / self.model_key
        if out_dir.exists():
            print(f"[skip] {out_dir} already exists")
            return
        out_dir.mkdir(parents=True, exist_ok=False)

        self._load_model()

        # accumulation buffers for chunk saving
        buf_hidden = []
        buf_target = []
        buf_sent = []
        buf_tok_pos = []
        buf_hid_pos = []

        chunk_idx = 0        # index of the next chunk file to save
        sentence_offset = 0  # global sentence index offset across batches

        for batch_start in tqdm(range(0, len(texts), self.tokenizer_batch_size), desc=f"NTP [{self.model_key}]", dynamic_ncols=True):
            batch_texts = texts[batch_start: batch_start + self.tokenizer_batch_size]

            # tokenize batch
            input_ids, attention_mask = self._tokenize(batch_texts)

            # generate (target, h_{t-1}) pairs
            target_ids, sent_ids, tok_positions, hid_positions, batch_indices = self._make_examples(
                input_ids, attention_mask, sentence_offset,
            )

            # single forward pass per batch -> extract hidden at h_{t-1} positions
            if hid_positions:
                hidden = self._forward_batch(input_ids, attention_mask, batch_indices, hid_positions)

                buf_hidden.append(hidden)
                buf_target.extend(target_ids)
                buf_sent.extend(sent_ids)
                buf_tok_pos.extend(tok_positions)
                buf_hid_pos.extend(hid_positions)

            sentence_offset += len(batch_texts)

            # flush to disk when accumulated samples reach chunk_size
            total = sum(h.shape[0] for h in buf_hidden)
            if total >= self.chunk_size:
                self._flush_chunk(buf_hidden, buf_target, buf_sent, buf_tok_pos, buf_hid_pos, out_dir, chunk_idx)
                chunk_idx += 1
                buf_hidden, buf_target, buf_sent, buf_tok_pos, buf_hid_pos = [], [], [], [], []

        # save any remaining samples as the final chunk
        if buf_hidden:
            self._flush_chunk(buf_hidden, buf_target, buf_sent, buf_tok_pos, buf_hid_pos, out_dir, chunk_idx)

        del self.model
        self.model = None
        torch.cuda.empty_cache()

    # concatenates buffers and saves one chunk .pt file
    def _flush_chunk(self, buf_hidden, buf_target, buf_sent, buf_tok_pos, buf_hid_pos, out_dir, chunk_idx):
        hidden = torch.cat(buf_hidden, dim=0)  # [N, hidden_dim]
        data = {
            "hidden": hidden.float(),
            "target_token_ids": torch.tensor(buf_target, dtype=torch.long),
            "sentence_ids": torch.tensor(buf_sent, dtype=torch.long),
            "token_positions": torch.tensor(buf_tok_pos, dtype=torch.long),
            "hidden_positions": torch.tensor(buf_hid_pos, dtype=torch.long),
            "model_name": self.model_hf_id,
            "backbone_type": "ntp",
        }
        out_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
        self._save_chunk(data, out_path)

    @torch.no_grad()
    def extract_to_memory(self, texts: list[str]) -> torch.Tensor:
        """Extracts NTP hidden vectors into memory without saving to disk.

        Returns:
            hidden: [N, hidden_dim] float tensor over all non-special tokens.
        """
        if self.model is None:
            self._load_model()

        all_hidden = []
        for batch_start in range(0, len(texts), self.tokenizer_batch_size):
            batch_texts = texts[batch_start: batch_start + self.tokenizer_batch_size]
            input_ids, attention_mask = self._tokenize(batch_texts)
            _, _, _, hid_positions, batch_indices = self._make_examples(
                input_ids, attention_mask, sentence_offset=0,
            )
            if hid_positions:
                hidden = self._forward_batch(input_ids, attention_mask, batch_indices, hid_positions)
                all_hidden.append(hidden.float())

        return torch.cat(all_hidden, dim=0) if all_hidden else torch.empty(0)
