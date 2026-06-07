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

```bash
# (gutenberg, mlm)
python scripts/train.py --dataset gutenberg --backbone mlm --model bert
python scripts/train.py --dataset gutenberg --backbone mlm --model roberta
python scripts/train.py --dataset gutenberg --backbone mlm --model modernbert

# (gutenberg, ntp)
python scripts/train.py --dataset gutenberg --backbone ntp --model llama3_3b
python scripts/train.py --dataset gutenberg --backbone ntp --model llama31_8b
python scripts/train.py --dataset gutenberg --backbone ntp --model opt_1.3b

# (opensubtitles, mlm)
python scripts/train.py --dataset opensubtitles --backbone mlm --model bert
python scripts/train.py --dataset opensubtitles --backbone mlm --model roberta
python scripts/train.py --dataset opensubtitles --backbone mlm --model modernbert

# (opensubtitles, ntp)
python scripts/train.py --dataset opensubtitles --backbone ntp --model llama3_3b
python scripts/train.py --dataset opensubtitles --backbone ntp --model llama31_8b
python scripts/train.py --dataset opensubtitles --backbone ntp --model opt_1.3b
```
