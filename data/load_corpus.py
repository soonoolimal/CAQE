"""
Corpus loaders for Gutenberg and OpenSubtitles datasets.
Downloads from HuggingFace and caches locally as JSONL.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from datasets import load_dataset

from data.preprocess_utils import load_jsonl, preprocess_texts, save_jsonl


class BaseCorpusLoader(ABC):
    """Base class for corpus loaders.

    Loads from a local JSONL if available,
    otherwise downloads from HuggingFace, preprocesses, saves to JSONL, and returns the result.
    """
    def __init__(
        self,
        data_config: dict,       # data.yaml
        preprocess_config: dict  # preprocess.yaml
    ):
        self.data_config = data_config

        output_dir = Path(preprocess_config["output_dir"])

        # save path: {output_dir}/{processed_data_dir}/{dataset_name}.jsonl
        self.jsonl_path = output_dir / preprocess_config["processed_data_dir"] / f"{self._dataset_name()}.jsonl"

    # returns the dataset identifier used for the local JSONL filename
    @abstractmethod
    def _dataset_name(self) -> str:
        pass

    # downloads from HuggingFace, preprocesses, and returns a list of sentences
    @abstractmethod
    def _download_and_preprocess(self) -> list[str]:
        pass

    def load(self) -> list[str]:
        # load from local JSONL if it already exists
        if self.jsonl_path.exists():
            print(f"[{self._dataset_name()}] Local file found. Loading: {self.jsonl_path}")
            return load_jsonl(self.jsonl_path)

        # no local JSONL: download from HuggingFace, preprocess, and save
        print(f"[{self._dataset_name()}] No local file found. Downloading from HuggingFace")
        texts = self._download_and_preprocess()
        save_jsonl(texts, self.jsonl_path)
        return texts


class GutenbergLoader(BaseCorpusLoader):
    """Corpus loader for willwade/Gutenberg-dialog-en.

    Each row is a single utterance, and empty rows mark conversation boundaries.
    Since the dataset is low-noise, no extra filtering is applied beyond standard preprocessing.
    """
    def __init__(self, data_config: dict, preprocess_config: dict):
        super().__init__(data_config, preprocess_config)

    def _dataset_name(self) -> str:
        return "gutenberg"

    def _download_and_preprocess(self) -> list[str]:
        cfg = self.data_config

        dataset = load_dataset(cfg["hf_id"], split=cfg["split"])
        raw_texts = dataset[cfg["text_column"]]

        # filter out empty rows (conversation boundaries)
        raw_texts = [t for t in raw_texts if t.strip()]
        print(f"After empty row filtering: {len(raw_texts):,}")

        # split=True: each utterance may contain multiple sentences
        return preprocess_texts(raw_texts, split=True)


class OpenSubtitlesLoader(BaseCorpusLoader):
    """Corpus loader for Helsinki-NLP/open_subtitles.

    Extracts only the English(lang1) side from the parallel corpus and applies noise filtering.
    Since subtitles are already sentence-level, only whitespace cleaning is applied without pysbd splitting.
    """
    def __init__(self, data_config: dict, preprocess_config: dict):
        super().__init__(data_config, preprocess_config)

    def _dataset_name(self) -> str:
        return "opensubtitles"

    # OpenSubtitles is documented as high-noise
    def _is_noisy(self, text: str) -> bool:
        cfg = self.data_config["noise_filter"]

        # too short: likely a non-sentence fragment (e.g., "OK", "Yeah")
        if len(text) < cfg["min_chars"]:
            return True

        # too long: likely a subtitle merge error
        if len(text) > cfg["max_chars"]:
            return True

        # too few alphabetic characters: likely noise (symbols, numbers, encoding errors)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < cfg["min_alpha_ratio"]:
            return True

        return False

    def _download_and_preprocess(self) -> list[str]:
        cfg = self.data_config

        # language pair must be specified to load a parallel corpus
        dataset = load_dataset(
            cfg["hf_id"],
            lang1=cfg["lang1"],
            lang2=cfg["lang2"],      # dummy
            split="train",           # only train split is available for OpenSubtitles
            trust_remote_code=True,  # required: dataset uses a custom loading script
        )

        # extract only the lang1(English) side from each translation pair
        raw_texts = [item["translation"][cfg["lang1"]] for item in dataset]
        print(f"After language extraction: {len(raw_texts):,}")

        # apply noise filters
        raw_texts = [t for t in raw_texts if not self._is_noisy(t)]
        print(f"After noise filtering: {len(raw_texts):,}")

        # apply whitespace cleaning only
        return preprocess_texts(raw_texts, split=False)  # split=False: subtitles are already sentence-level
