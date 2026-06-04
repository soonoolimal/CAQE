"""
Text preprocessing utilities: cleaning, sentence splitting, and JSONL I/O.
"""

import json
import re
from pathlib import Path
from typing import Iterable

import pysbd

# initialize pysbd English sentence segmenter once at module load
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


# collapse newlines, tabs, and repeated spaces
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# split text into sentences using pysbd (handles abbreviations, punctuation, etc.)
def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    return list(_SEGMENTER.segment(text))


# clean and optionally split an iterable of texts into a flat list of sentences
def preprocess_texts(
    texts: Iterable[str],
    split: bool,
) -> list[str]:
    processed = []

    for text in texts:
        text = clean_text(text)

        if not text:
            continue

        # split paragraph and utterance-level text into individual sentences
        if split:
            for sent in split_sentences(text):
                sent = clean_text(sent)
                if sent:
                    processed.append(sent)
        # add as-is if already sentence-level (e.g., subtitles)
        else:
            processed.append(text)

    print(f"Preprocessing complete: {len(processed):,} sentences")
    return processed


# write texts to a JSONL file, one {"text": "..."} object per line
def save_jsonl(texts: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            text = clean_text(text)

            if not text:
                continue

            # save one JSON object per line: {"text": "..."}
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved: {path} ({count:,} lines)")


# load texts from a JSONL file and return as a list of strings
def load_jsonl(path: Path) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # parse each line and extract the text field
            obj = json.loads(line)
            texts.append(obj["text"])

    return texts
