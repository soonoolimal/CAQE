# CAQE

## Preprocess

```bash
# MLM (BERT-family)
python scripts/preprocess.py --dataset gutenberg --backbone mlm
python scripts/preprocess.py --dataset opensubtitles --backbone mlm

# NTP (GPT-family)
python scripts/preprocess.py --dataset gutenberg --backbone ntp
python scripts/preprocess.py --dataset opensubtitles --backbone ntp
```

## Train

Trains on the combined hidden vectors from all datasets (gutenberg + opensubtitles).

```bash
# mlm
python scripts/train.py --backbone mlm --model bert
python scripts/train.py --backbone mlm --model roberta
python scripts/train.py --backbone mlm --model modernbert

# ntp
python scripts/train.py --backbone ntp --model llama3_3b
python scripts/train.py --backbone ntp --model llama31_8b
python scripts/train.py --backbone ntp --model opt_1.3b
```
