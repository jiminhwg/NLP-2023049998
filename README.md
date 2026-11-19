# Transferability of LLM-generated adversarial examples across BERT-family models

The experiment compares a word-level attack (TextFooler) with iterative LLM sentence rewriting on SST-2 and measures how often an adversarial example built against one classifier also fools the other two.

Repository: https://github.com/jiminhwg/NLP-2023049998

## Contents

```
main.ipynb                   attacks and all analyses; needs a GPU (run on Kaggle, T4)
reproduce_tables.py          recomputes the result tables from the saved results; CPU only
requirements.txt             packages for reproduce_tables.py
README.md
results/                     saved outputs of the reported run (about 0.7 MB in total)
  main_textfooler.jsonl        one record per (source classifier, sentence) for TextFooler
  main_llm_v3.jsonl            same for LLM rewriting, including the kept rewrite and its query count
  judge_cache.json             sentiment judgments for every text the judge saw
  main_transfer_table.csv      per-pair transfer table (written by the notebook, Section 5.2)
  handcheck.csv                sampled examples with method and source (not blind)
  handcheck_blind.csv          the same sample, shuffled, without the method; contains my labels
  handcheck_key.csv            method and source for each row of handcheck_blind.csv
  random_examples.csv          seeded random sample of valid and failed attacks (Analysis)
```

Model checkpoints are not included; they are downloaded from Hugging Face at run time.

## Environment

**Tables script (`reproduce_tables.py`).** Python 3 with `pandas` and `numpy`. It was tested in a clean directory with Python 3.12.3, pandas 3.0.6 and numpy 2.5.4. That is not the environment of the reported run.

**Full pipeline (notebook).** Kaggle notebook with Internet and a GPU (Tesla T4), Python 3.13. The install cell has no version pins, and the package versions of the reported run were not recorded. `[TODO: add package versions here if you can obtain them, and say how]`

Models and data come from Hugging Face and are not pinned to a revision:

| Role | Name |
|---|---|
| Classifier | `textattack/bert-base-uncased-SST-2` |
| Classifier | `textattack/roberta-base-SST-2` |
| Classifier | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Rewriting model and sentiment judge | `Qwen/Qwen2.5-3B-Instruct` (fp16) |
| Similarity encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| NLI check | `roberta-large-mnli` |
| Data | GLUE SST-2 (`load_dataset('glue', 'sst2')`), validation split |

Total run time of the reported run (both attacks, all three sources): 3 h 14 min 45 s.

## Reproducing the main tables

### From the saved results (no GPU)

```
pip install -r requirements.txt
python reproduce_tables.py --results_dir results/ --out_csv per_pair_table.csv --examples_csv random_examples_regenerated.csv
```

The script prints one block per result. The first column gives the table or section of the paper (fill in the numbers of the final version).

| Paper table / section | Block printed by the script |
|---|---|
| `[TODO]` Source success; appendix table on the success funnel | `Source success` |
| `[TODO]` Per-pair conditional transfer and figure; per-pair unconditional rates | `Per-pair transfer`; the CSV also has the unconditional columns |
| `[TODO]` Pooled transfer, similarity 0.80 and 0.85 (exploratory) | `Pooled transfer` |
| `[TODO]` Queries | `Classifier queries` |
| `[TODO]` Edit-size statistics (Analysis) | `Edit size of valid examples` |
| `[TODO]` Judge removal, concentration, shared-text decomposition, round of success (Analysis, exploratory) | the four `Exploratory:` blocks |
| `[TODO]` Hand-check proportions with Wilson intervals | `Hand-check counts`; the counts are typed into the script from the notebook output |
| `[TODO]` Random examples | `--examples_csv` |

The per-pair CSV written by the script is identical to `results/main_transfer_table.csv`. The terciles in the edit-size block can differ by one example from the notebook's printout, depending on the pandas version.

### Only in the notebook (needs the GPU and the models)

| Result | Notebook section |
|---|---|
| Clean accuracy, confusion matrices | 2.1 |
| Judge agreement with gold labels on the 200 sentences | 3.2 |
| TextFooler and LLM attacks (hours) | 4.1, 4.2 |
| NLI check and NLI-filtered pooled comparison | 6.2, 6.3 |
| Blind hand labeling (interactive) and its results | 6.1, 6.4 |

To rerun the notebook: upload it to a Kaggle notebook with Internet and the T4 GPU on, run the cells from top to bottom, and restart the session after the install cell if asked. The settings are in Section 1.2 (`mode = 'main'`, 200 validation sentences). Attack results are appended after every sentence and finished sentences are skipped on a re-run, so an interrupted run can be resumed.

Section 5.3 regenerates the three `handcheck*.csv` files and overwrites any saved labels. When rerunning, copy `results/handcheck_blind.csv` and `results/handcheck_key.csv` back after running 5.3 and skip 6.1.

## Random seeds

| Component | Seed |
|---|---|
| Global (`random`, NumPy, PyTorch) | 42 (`CFG['seed']`) |
| Sentence selection | `RandomState(42)` |
| LLM sampling | 42 + sentence id, set before each (sentence, source) attack. The same seed is used for all three sources |
| Bootstrap, per-pair table | `RandomState(0)`, 2000 resamples |
| Bootstrap, pooled and NLI-subset comparisons | `RandomState(1)`, 2000 resamples |
| Bootstrap, judge-removal comparison (script) | `RandomState(2)`, 2000 resamples |
| Bootstrap, unique-text comparison (script) | `RandomState(3)`, 2000 resamples |
| Hand-check sample and its shuffle | `random_state=0` |
| Random examples (`random_examples.csv`) | `random_state=0` |

## Known sources of variation

- LLM sampling is seeded, but sampled text can differ across GPU types and library versions. `results/main_llm_v3.jsonl` holds the rewrites behind every reported number, so the transfer tables and analyses can be recomputed from it without calling the LLM.
- Models and data are not pinned to a revision, and package versions were not recorded.
- The clean accuracies, judge agreement and NLI numbers in the paper come from the notebook, not from the script.
- A fresh full run is expected to give similar, not identical, LLM numbers. I did not check how deterministic TextFooler is across TextAttack versions.

## Fresh-machine check

- [ ] `reproduce_tables.py` on a machine without my files: `[TODO: date, OS, Python, pandas, numpy, result]`
- [ ] Notebook in a fresh Kaggle session with `mode = 'pilot'` (installs and runs through Section 5.2): `[TODO: date, result, problems]`

Run the pilot check in a session that does not contain the real results folder, because Section 5.3 writes the hand-check files without a mode prefix.
