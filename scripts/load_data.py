from pathlib import Path

import yaml

from data.extractor import MLMHiddenExtractor, NTPHiddenExtractor
from data.preprocessor import GutenbergPreprocessor, OpenSubtitlesPreprocessor


def load_data():
    cfg_path = Path(__file__).parents[1] / "configs" / "data.yaml"
    with open(cfg_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    texts = GutenbergPreprocessor(data_cfg).load() + OpenSubtitlesPreprocessor(data_cfg).load()

    for model_key, backbone_cfg in data_cfg["backbone"].items():
        print(f"\n{'='*60}")
        print(f"[{model_key}] total sentences: {len(texts):,}")
        print(f"{'='*60}")

        if backbone_cfg["type"] == "mlm":
            extractor = MLMHiddenExtractor(model_key, data_cfg)
        else:
            extractor = NTPHiddenExtractor(model_key, data_cfg)

        extractor.extract(texts)
