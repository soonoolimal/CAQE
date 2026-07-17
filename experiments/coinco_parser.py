"""
CoinCo dataset parser.

File format (one line per annotated content token):
    lemma.POS | sentence | sub1::freq1 sub2::freq2 ...
    E.g., happy.JJ | I am happy to see you | glad::3 pleased::2 content::1
"""

from dataclasses import dataclass
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer

_NLTK_DIR = Path(__file__).parents[1] / "data" / "experiments" / "raw"
nltk.data.path.insert(0, str(_NLTK_DIR))

try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    nltk.download("wordnet", download_dir=str(_NLTK_DIR))

_SURROUNDING_PUNCT = ".,!?;:\"'()[]"

# Penn Treebank POS tags -> WordNet POS tags for lemmatizer
_POS_TO_WN = {
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
}


@dataclass
class CoinCoToken:
    target_word: str
    pos_tag: str
    sentence: str
    substitutes: list[tuple[str, int]]  # sorted by count desc (i.e., top-1 is [0])


# "glad::3 pleased::2 content::1 assent to::1" -> [("glad", 3), ("pleased", 2), ("content", 1), ("assent to", 1)]
def _parse_substitutes(subs_str: str) -> list[tuple[str, int]]:
    parts = subs_str.strip().split("::")
    result = []
    word = parts[0].strip()

    # each part: "count next_word" (next_word may contain spaces)
    for part in parts[1:]:
        tokens = part.split(None, 1)
        result.append((word, int(tokens[0])))
        word = tokens[1].strip() if len(tokens) > 1 else ""

    result.sort(key=lambda x: x[1], reverse=True)  # top-1 substitute

    return result


# "happy.JJ | I am happy to see you | glad::3 pleased::2"
# -> CoinCoToken(target_word="happy", pos_tag="JJ", sentence="I am happy to see you", substitutes=[("glad", 3), ...])
def parse_coinco(path: Path) -> list[CoinCoToken]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" | ")
            if len(parts) != 3:
                continue

            target_pos, sentence, subs_str = parts
            dot = target_pos.rfind(".")
            target_word = target_pos[:dot]
            pos_tag = target_pos[dot + 1 :]

            substitutes = _parse_substitutes(subs_str)
            if not substitutes:
                continue

            entries.append(
                CoinCoToken(
                    target_word=target_word,
                    pos_tag=pos_tag,
                    sentence=sentence,
                    substitutes=substitutes,
                )
            )

    return entries


# ("The companies have merged.", "merge", "VBN") -> 3  ("merged" matches lemma "merge")
# ("The dog runs fast.", "cat", "NN") -> None  (not found)
def find_target_position(sentence: str, target_word: str, pos_tag: str) -> int | None:
    lemmatizer = WordNetLemmatizer()  # reduces inflected surface forms to base lemma (e.g., "merged" -> "merge")
    wn_pos = _POS_TO_WN.get(pos_tag, "n")  # fallback to "n" for unknown tags (function words already filtered upstream)
    target_lower = target_word.lower()
    words = sentence.split()
    matches = []
    for i, word in enumerate(words):
        surface = word.lower().strip(_SURROUNDING_PUNCT)  # normalize: lowercase, strip surrounding punctuation
        if surface == target_lower or lemmatizer.lemmatize(surface, pos=wn_pos) == target_lower:
            matches.append(i)
    return matches[0] if len(matches) == 1 else None


# fraction of entries whose target word cannot be located in its sentence (backbone-independent skip)
def not_found_rate(entries: list[CoinCoToken]) -> float:
    n_not_found = sum(find_target_position(e.sentence, e.target_word, e.pos_tag) is None for e in entries)
    return n_not_found / len(entries)
