"""
Downloads coinco.xml.gz from IMS Stuttgart and converts it to data/experiments/coinco.txt.

Input XML structure:
    <document>
        <sent>
            <targetsentence>A mission to end a war</targetsentence>
            <tokens>
                <token id="4" wordform="mission" lemma="mission" posMASC="NN">
                    <substitutions>
                        <subst lemma="goal" freq="2"/>
                    </substitutions>
                </token>
                <token id="XXX" wordform="a" lemma="a" posMASC="XXX"/>
            </tokens>
        </sent>
    </document>
"""

import gzip
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

_ROOT = Path(__file__).parents[1]
_EXP_CFG_PATH = _ROOT / "configs" / "experiments.yaml"


def load_coinco_cfg() -> tuple[str, Path]:
    with open(_EXP_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    coinco = cfg["datasets"]["coinco"]
    return coinco["url"], _ROOT / coinco["path"]


def download_xml(url: str) -> bytes:
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    print(f"Downloaded {len(data) / 1024:.1f} KB")
    return data


def convert_xml_to_txt(xml_bytes: bytes, out_path: Path):
    xml_text = gzip.decompress(xml_bytes)
    root = ET.fromstring(xml_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        # per-sentence: extract target sentence text
        for sent in root.iter("sent"):
            sentence_el = sent.find("targetsentence")
            if sentence_el is None or not sentence_el.text:  # text stored with raw newlines in XML, skip if absent
                continue
            sentence = " ".join(sentence_el.text.split())  # collapse newlines and whitespace from raw XML text

            tokens_el = sent.find("tokens")
            if tokens_el is None:
                continue

            # per-token: extract lemma, pos, substitutes and write one line
            for token in tokens_el.iter("token"):
                # skip function words and non-content tokens
                if token.get("id") == "XXX":
                    continue

                # skip tokens with missing attributes
                lemma = token.get("lemma", "")
                pos = token.get("posMASC", "")
                if not lemma or not pos:
                    continue

                # skip tokens with no substitutions element
                subs_el = token.find("substitutions")
                if subs_el is None:
                    continue

                # skip tokens with no valid substitutes
                substitutes = [
                    (sub.get("lemma", ""), int(sub.get("freq", "0")))
                    for sub in subs_el.iter("subst")
                    if sub.get("lemma") and sub.get("freq", "0") != "0"
                ]
                if not substitutes:
                    continue

                subs_str = " ".join(f"{w}::{c}" for w, c in substitutes)
                f.write(f"{lemma}.{pos} | {sentence} | {subs_str}\n")
                n_written += 1

    print(f"Written {n_written} entries -> {out_path}")


def load_coinco():
    url, out_path = load_coinco_cfg()

    if out_path.exists():
        print(f"Already exists: {out_path}")
        return

    xml_bytes = download_xml(url)
    convert_xml_to_txt(xml_bytes, out_path)
