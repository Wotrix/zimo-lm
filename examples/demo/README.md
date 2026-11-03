# Bundled demo model

A 4.31M parameter character-level GPT-2 trained on 30M characters of
Chinese dialogue (LCCC subset), val PPL 82.3.

- `model_best.pt` - all-in-one checkpoint (weights + vocab), 17 MB
- `vocab.txt` - 4250 characters + `<unk>`

The Studio loads this automatically when `runs/` is empty, so a fresh
clone can score text and try the playground before training anything.
It is a thin demo, not a production model: dialogue domain, 4M params.
