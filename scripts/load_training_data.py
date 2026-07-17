from pathlib import Path

import yaml

from train.extractor import MLMHiddenExtractor, NTPHiddenExtractor
from train.preprocessor import GutenbergPreprocessor, OpenSubtitlesPreprocessor

_DATA_CFG = Path(__file__).parents[1] / "configs" / "data.yaml"


def load_training_data():
    with open(_DATA_CFG) as f:
        data_cfg = yaml.safe_load(f)

    texts = GutenbergPreprocessor(data_cfg).load() + OpenSubtitlesPreprocessor(data_cfg).load()

    for backbone_key, backbone_cfg in data_cfg["backbone"].items():
        print(f"\n{'=' * 60}")
        print(f"[{backbone_key}] total sentences: {len(texts):,}")
        print(f"{'=' * 60}")

        if backbone_cfg["type"] == "mlm":
            extractor = MLMHiddenExtractor(backbone_key, data_cfg)
        else:
            extractor = NTPHiddenExtractor(backbone_key, data_cfg)

        extractor.extract(texts)
