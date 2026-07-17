"""
Parses Natural Stories Corpus from following sources:
    - parses/penn/all-parses-aligned.txt.penn: sentence boundaries + POS tags
    - naturalstories_RTS/all_stories.tok: surface forms per (item, zone)
    - naturalstories_RTS/processed_RTs.tsv: per-subject per-word RT

Penn leaf format: (POS surface/item.zone[.sub])
    - sub-index (.1, .2, ...) means one all_stories.tok zone was split into
        multiple parse tokens (e.g., "England," -> NNP England/1.10.1 + , ,/1.10.2).
    - For POS mapping, only sub-index .1 (or no sub-index) is the main content token.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_NS_ROOT = Path(__file__).parents[1] / "data" / "experiments" / "raw" / "natural_stories" / "naturalstories-master"
_PENN_PATH = _NS_ROOT / "parses" / "penn" / "all-parses-aligned.txt.penn"
_TOK_PATH = _NS_ROOT / "naturalstories_RTS" / "all_stories.tok"
_RT_PATH = _NS_ROOT / "naturalstories_RTS" / "processed_RTs.tsv"

# Penn Treebank POS sets
_FUNC_POS = frozenset({"CC", "IN", "PRP", "PRP$", "WP", "WP$"})
_PUNCT_POS = frozenset({",", ".", ":", "``", "''", "-LRB-", "-RRB-", "#", "$"})

# matches leaf nodes with zone: (POS surface/item.zone) or (POS surface/item.zone.sub)
_LEAF_RE = re.compile(r"\(([^\s()]+)\s+[^\s/()]+/(\d+)\.(\d+)(?:\.(\d+))?\)")


@dataclass
class NSWord:
    zone: int
    surface: str  # as in all_stories.tok (e.g., "England,")
    pos: str  # POS of the main sub-token (.1 or no sub-index)
    has_punct: bool  # True if any sub-token of this zone has punctuation POS


@dataclass
class NSSentence:
    story_id: int
    sent_id: int  # 0-based index within story
    words: list[NSWord]
    is_story_start: bool = False  # first sentence of its story; set by parse_natural_stories
    is_story_end: bool = False  # last sentence of its story; set by parse_natural_stories

    @property
    def text(self) -> str:
        return " ".join(w.surface for w in self.words)

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def zone_range(self) -> tuple[int, int]:
        zones = [w.zone for w in self.words]
        return min(zones), max(zones)

    # sentence-level ratios: fraction of words carrying the feature (binary "any" is ~constant, see reading_time.md)
    def x_punct(self) -> float:
        return sum(1 for w in self.words if w.has_punct) / self.n_words

    def x_func(self) -> float:
        return sum(1 for w in self.words if w.pos in _FUNC_POS) / self.n_words

    # story-boundary binary indicators (sentence-boundary sos/eos is degenerate at sentence level, see reading_time.md)
    def x_sos(self) -> float:
        return float(self.is_story_start)

    def x_eos(self) -> float:
        return float(self.is_story_end)


# (item, zone) -> surface form
def _load_surfaces(tok_path: Path) -> dict[tuple[int, int], str]:
    surfaces = {}
    with open(tok_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, zone, item = parts
            surfaces[(int(item), int(zone))] = word
    return surfaces


# parse Penn Treebank blocks into NSSentence objects using surface forms from all_stories.tok
def _parse_penn(penn_path: Path, surfaces: dict[tuple[int, int], str]) -> list[NSSentence]:
    text = penn_path.read_text(encoding="utf-8")
    blocks = re.split(r"(?=\(ROOT)", text)
    blocks = [b.strip() for b in blocks if b.strip().startswith("(ROOT")]

    sentences: list[NSSentence] = []
    sent_counter: dict[int, int] = {}

    for block in blocks:
        leaves = _LEAF_RE.findall(block)
        # leaves: [(pos, item_str, zone_str, sub_str_or_empty), ...]

        if not leaves:
            continue

        # all leaves in one ROOT should share the same story/item
        items = {int(item) for _, item, _, _ in leaves}
        if len(items) != 1:
            continue
        story_id = items.pop()

        # for each zone: track main POS (sub=1 or no sub) and punctuation presence (any sub)
        zone_to_pos: dict[int, str] = {}
        zone_has_punct: dict[int, bool] = {}
        for pos, _, zone_str, sub_str in leaves:
            zone = int(zone_str)
            if zone not in zone_to_pos or sub_str in ("", "1"):
                zone_to_pos[zone] = pos
            if pos in _PUNCT_POS:
                zone_has_punct[zone] = True

        # build words in zone order
        words = []
        for zone in sorted(zone_to_pos):
            surface = surfaces.get((story_id, zone), "")
            if not surface:
                continue
            words.append(
                NSWord(zone=zone, surface=surface, pos=zone_to_pos[zone], has_punct=zone_has_punct.get(zone, False))
            )

        if not words:
            continue

        sent_id = sent_counter.get(story_id, 0)
        sent_counter[story_id] = sent_id + 1
        # one ROOT block = one unique sentence
        sentences.append(NSSentence(story_id=story_id, sent_id=sent_id, words=words))

    return sentences


# mark the first (sent_id 0) and last sentence of each story for story-boundary sos/eos
def _mark_story_boundaries(sentences: list[NSSentence]) -> None:
    last_sent_id: dict[int, int] = {}
    for s in sentences:
        last_sent_id[s.story_id] = max(last_sent_id.get(s.story_id, 0), s.sent_id)
    for s in sentences:
        s.is_story_start = s.sent_id == 0
        s.is_story_end = s.sent_id == last_sent_id[s.story_id]


def parse_natural_stories(penn_path: Path = _PENN_PATH, tok_path: Path = _TOK_PATH) -> list[NSSentence]:
    surfaces = _load_surfaces(tok_path)
    sentences = _parse_penn(penn_path, surfaces)
    _mark_story_boundaries(sentences)
    return sentences


# load per-subject per-word RT filtered by [rt_min, rt_max]
def load_rts(rt_path: Path = _RT_PATH, rt_min: float = 50.0, rt_max: float = 3000.0) -> pd.DataFrame:
    df = pd.read_csv(rt_path, sep="\t", usecols=["WorkerId", "item", "zone", "RT"])
    df = df.rename(columns={"WorkerId": "subject_id", "RT": "rt"})
    df = df[(df["rt"] >= rt_min) & (df["rt"] <= rt_max)].reset_index(drop=True)
    return df
