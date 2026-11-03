# Model card: zimo-lm (reference configuration)

This card describes the **example configuration** used by the training commands
in the README.  It is not a released checkpoint; train your own model and add
its evaluation results here.

## Model details

| Item | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-2 style: pre-LN, causal self-attention, GELU MLP) |
| Parameters | 4,314,240 for 4,251 tokens, 6 layers, 192 dim, 4 heads, block 64 |
| Tokenizer | Character-level; special token `<unk>` = id 0 |
| Context | 64 tokens |
| Objective | Next-token cross-entropy |
| Output | Per-token log-probability; sequence score is the mean over the candidate |
| Initialisation | Random; no pretrained GPT-2 weights |
| Precision | fp32 training; int8 export is not included |

This is the **reference configuration**, not the maximum.  The lower/upper
bounds for this trainer are documented in the README
[Scope section](README.md#scope-lower-bound-and-upper-bound): roughly a 1M-param
/ 10M-token floor for a meaningful LM, 4M / 40–80M tokens for a rerank baseline,
~25M / 250–500M tokens as the practical ceiling for LCCC, and 50M+ requiring
more data and a larger GPU.  100M+ is out of scope for this reference trainer.

## Training data

The example commands use a sample of **LCCC** (Chinese conversational corpus):
base + large dialogues, converted to character-token lines.  The corpus is not
included in this repository.  See [DATA_CARD.md](DATA_CARD.md) for provenance,
license, and privacy notes.  The exact token budget is controlled by
`--max-chars` / `--tokens`; the README example uses about 30M characters.

## Intended use

- Language-model scoring of candidate Chinese text for IME reranking.
- A small reproducible baseline for character-level Chinese LM experiments.
- Offline analysis of context/naturalness signals before integrating a runtime.

## Out of scope

- Pinyin-to-Hanzi decoding or candidate generation.
- Chat, instruction following, question answering, or factual retrieval.
- Any high-stakes decision; the model is small, data-limited, and not audited
  for safety or bias.

## Evaluation

- Training curves: loss and perplexity on a held-out tail split (see
  `metrics.jsonl`).
- Task evaluation: use `eval_gpt_rerank.py` with a held-out candidate set and
  report top-1 before/after, coverage, and per-category results.
- **Do not use perplexity as the acceptance metric.**  A lower PPL does not
  imply better candidate ranking.

## Limitations

- Trained on conversational Chinese; formal, technical, and mixed English text
  may be out of distribution.
- Character-level, no word boundaries; punctuation is a token.
- Tiny context (64 tokens) and a 4.3M parameter budget.
- No pinyin or keystroke information.
- No safety alignment, no factuality guarantee, no personalisation.

## License

Code: MIT.  Weights are a separate artifact.  If you release weights trained
on LCCC, include the upstream attribution and comply with the upstream terms;
do not release weights trained on data you cannot redistribute.
