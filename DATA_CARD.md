# Data card

## LCCC (example training corpus)

| Item | Value |
|---|---|
| Name | LCCC (Large-scale Chinese Conversation Corpus) |
| Source | `thu-coai/CDial-GPT` and the LCCC distribution referenced there |
| Upstream license | MIT (verify the authoritative upstream terms before redistribution) |
| Scale | LCCC-base: ~6.8M dialogues; LCCC-large: ~12M dialogues; the two are largely disjoint |
| Text volume | ~612M non-whitespace characters (base + large, measured locally) |
| Domain | Chinese social-media / conversational dialogue |
| Format | JSONL; one JSON array of utterance strings per line |

LCCC is **not included** in this repository.  Download it from the upstream
project, then convert it:

```bash
python prepare_corpus.py --input lccc_base_train.jsonl.gz \
  --output corpus.txt --max-chars 15000000
python prepare_corpus.py --input lccc_large.jsonl.gz \
  --output corpus.txt --max-chars 15000000 --append
```

## Preprocessing

- Each utterance is stripped of whitespace and re-split into characters.
- Punctuation is kept as its own token.
- One utterance per line; the trainer flattens lines into a token stream.
- Dialogue/turn boundaries are not marked with a special token in the reference
  trainer.  If you need explicit boundaries, insert a reserved token and add it
  to the vocabulary.

## Privacy and redistribution

LCCC contains public social-media conversations.  Even when a corpus is
publicly available, review it for personal information before redistributing
the raw text.  The safest default is to publish the **preparation script**, not
the corpus.  Model weights trained on LCCC may still memorise fragments; do not
release weights without checking the upstream terms and your own privacy review.

## Other data

- `examples/sample.txt` is a tiny set of ordinary Chinese sentences authored for
  the smoke test.  It contains no third-party corpus.
- No IME dictionary, no proprietary word list, and no competitor model or
  weight is included.
