import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pysbd
from datasets import Dataset, load_dataset
from tqdm import tqdm


class BasePreprocessor(ABC):
    _segmenter = pysbd.Segmenter(language="en", clean=False)

    def __init__(self, data_cfg: dict):
        self.sample_size = data_cfg["sample_size"]
        self.seed = data_cfg["seed"]
        self.src_cfg = data_cfg["source"][self._name()]

        save_cfg = data_cfg["save"]
        self.save_path = Path(save_cfg["dirs"]["preprocessed"]) / f"{self._name()}.jsonl"

    @abstractmethod
    def _name(self) -> str:
        pass

    @abstractmethod
    def _download_and_preprocess(self) -> list[str]:
        pass

    def load(self) -> list[str]:
        """Load sentences from cached JSONL if available, otherwise download from HF, preprocess, and save to JSONL."""
        if self.save_path.exists():
            print(f"[{self._name()}] Loading from cache: {self.save_path}")
            return self._load_jsonl()

        print(f"[{self._name()}] No cache found, downloading from HuggingFace")
        texts = self._download_and_preprocess()
        self._save_jsonl(texts)
        return texts

    def _load_jsonl(self) -> list[str]:
        texts = []
        with open(self.save_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
        print(f"[{self._name()}] Loaded {len(texts):,} sentences")
        return texts

    # writes one {"text": "..."} JSON object per line
    def _save_jsonl(self, texts: list[str]):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(self.save_path, "w", encoding="utf-8") as f:
            for text in texts:
                if not text:
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
        print(f"[{self._name()}] Saved {count:,} sentences -> {self.save_path}")

    # collapses consecutive whitespace (spaces, tabs, newlines) into a single space
    # and strips leading/trailing whitespace
    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _split_sentences(self, text: str) -> list[str]:
        return list(self._segmenter.segment(text))

    def _sample(self, dataset: Dataset) -> Dataset:
        original_size = len(dataset)
        if original_size > self.sample_size:
            dataset = dataset.shuffle(seed=self.seed).select(range(self.sample_size))
            print(f"[{self._name()}] Sampled {self.sample_size:,} / {original_size:,} rows")
        else:
            print(
                f"[{self._name()}] Dataset size ({original_size:,}) is smaller than "
                f"sample_size ({self.sample_size:,}), using all rows"
            )
        return dataset


class GutenbergPreprocessor(BasePreprocessor):
    def _name(self):
        return "gutenberg"

    def _download_and_preprocess(self):
        # download
        dataset = load_dataset(
            self.src_cfg["hf_id"],
            split="train",  # ignore valid/test since train/valid split is created from train
        )
        assert isinstance(dataset, Dataset)

        # sample randomly
        dataset = self._sample(dataset)

        # remove empty rows
        raw_texts = [t for t in dataset["text"] if t.strip()]
        print(f"[Gutenberg] Raw: {len(raw_texts):,} rows")

        # clean whitespaces and split into sentence-level
        sentences = []
        for text in tqdm(raw_texts, desc="[Gutenberg] Preprocessing"):
            text = self._clean(text)
            for sentence in self._split_sentences(text):
                sentence = self._clean(sentence)
                if sentence:
                    sentences.append(sentence)

        print(f"[Gutenberg] After preprocessing: {len(sentences):,} sentences")
        return sentences


class OpenSubtitlesPreprocessor(BasePreprocessor):
    def _name(self):
        return "opensubtitles"

    def _download_and_preprocess(self):
        # download
        dataset = load_dataset(
            self.src_cfg["hf_id"],
            lang1=self.src_cfg["lang1"],  # language to extract
            lang2=self.src_cfg["lang2"],  # dummy: partner language required to load the parallel corpus
            split="train",  # train/valid split is created from train
            trust_remote_code=True,
        )
        assert isinstance(dataset, Dataset)

        # sample randomly
        dataset = self._sample(dataset)

        raw_texts = [item["translation"][self.src_cfg["lang1"]] for item in dataset]

        # remove empty rows
        raw_texts = [t for t in raw_texts if t.strip()]
        print(f"[OpenSubtitles] Raw: {len(raw_texts):,} rows")

        # clean whitespaces and remove noisy sentences
        # no sentence-level split is needed since OpenSubtitles data is already in sentence-level
        sentences = []
        for text in tqdm(raw_texts, desc="[OpenSubtitles] Preprocessing"):
            text = self._clean(text)
            if self._is_noisy(text):
                continue
            sentences.append(text)

        print(f"[OpenSubtitles] After preprocessing: {len(sentences):,} sentences")
        return sentences

    def _is_noisy(self, text: str) -> bool:
        noise_cfg = self.src_cfg["noise_filter"]

        # removes sentences shorter than min_chars characters
        if len(text) < noise_cfg["min_chars"]:
            return True

        # filters out lines where alphabetic characters make up less than min_alpha_ratio of the total
        if sum(c.isalpha() for c in text) / len(text) < noise_cfg["min_alpha_ratio"]:
            return True

        return False
