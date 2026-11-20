# Transferability of LLM-generated adversarial examples across BERT-family models

The experiment compares a word-level attack (TextFooler) with iterative LLM sentence rewriting on SST-2 and measures how often an adversarial example built against one classifier also fools the other two.

Repository: https://github.com/jiminhwg/NLP-2023049998

## Contents

```
main.ipynb   attacks and all analyses; needs a GPU (run on Kaggle, Tesla T4)
reproduce_tables.py          recomputes the tables from the saved results; CPU only
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
```

Model checkpoints are not included; they are downloaded from Hugging Face at run time.

## Environment

**Tables script (`reproduce_tables.py`).** Python 3 with `pandas` and `numpy`. It was tested in a clean directory with Python 3.12.3, pandas 3.0.6 and numpy 2.5.4. That is not the environment of the reported run.

**Full pipeline (notebook).** Kaggle notebook with Internet and a GPU (Tesla T4), Python 3.13. The install cell has no version pins, and the package versions of the reported run were not recorded. `[TODO: add package versions here if you can obtain them, and say how]`

Models and data come from Hugging Face and are not pinned to a revision (the same list is Table 4 of the paper):

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

## Reproducing the tables in the draft

### From the saved results (no GPU)

```
pip install -r requirements.txt
python reproduce_tables.py --results_dir results/ --out_csv per_pair_table.csv --examples_csv random_examples_regenerated.csv
```

The script prints one block per result. Table numbers refer to the HW2 draft.

| Draft table | Content | Block printed by the script |
|---|---|---|
| Table 1 | Valid source attack rates and counts | `Source success`, last column (`valid`) |
| Table 2 | Conditional transfer rate for each of the six pairs, with the 95% interval of the LLM minus TextFooler difference | `Per-pair transfer`; the same table is written to `per_pair_table.csv` |
| Table 3 | Pooled conditional and unconditional rates at similarity 0.80 (exploratory) | `Pooled transfer`, threshold 0.8 |
| Table 5 (appendix) | TextFooler success before and after each validity check | `Source success`, columns `fooled`, `sim_ok`, `valid` |
| Table 6 (appendix) | Mean queries per attack and per valid attack | `Classifier queries` |
| Table 7 (appendix), rows with similarity 0.85 | Sensitivity analysis | `Pooled transfer`, threshold 0.85 |
| Table 7 (appendix), row "Unique-text examples only" | Conditional rates after removing examples whose exact text is valid on more than one source | `Exploratory: examples whose exact text is valid on another source`, line `unique-text only` |

### Only in the notebook (needs the GPU and the models)

| Draft content | Notebook section |
|---|---|
| Table 4 (appendix): clean accuracy of the three classifiers | 2.1 (checkpoint IDs: 1.3) |
| Judge agreement with the gold labels, 96.0% (Section 3.2) | 3.2 |
| TextFooler and LLM attacks (hours) | 4.1, 4.2 |
| Table 8 (appendix): NLI contradiction and entailment | 6.2 |
| Table 7 (appendix), rows "No NLI contradiction" and "Mutual entailment only" | 6.3 |
| Hand-check counts (Section 6.1): labeling and results | 6.1, 6.4 |

To rerun the notebook: upload it to a Kaggle notebook with Internet and the T4 GPU on, run the cells from top to bottom, and restart the session after the install cell if asked. The settings are in Section 1.2 (`mode = 'main'`, 200 validation sentences). Attack results are appended after every sentence and finished sentences are skipped on a re-run, so an interrupted run can be resumed.

Section 5.3 of the notebook regenerates the three `handcheck*.csv` files and overwrites any saved labels. When rerunning, copy `results/handcheck_blind.csv` and `results/handcheck_key.csv` back after running 5.3 and skip 6.1.

### Additional outputs not reported in the draft

The script also prints the mean edit ratio and similarity of valid examples (edit-size terciles), how many TextFooler examples the judge removes and how those transfer, how concentrated the valid examples are across sentences, the transfer rate by round of success, and Wilson intervals for the hand-check counts (typed in from the notebook output). `random_examples.csv` is a seeded random sample of 3 valid and 3 failed attacks per method.

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

## Design notes and known sources of variation

- **The LLM seed is shared across source classifiers.** The first prompt for a sentence does not depend on the source classifier, so with the same seed the first round of candidates is identical for all three sources, and later rounds often coincide. As a result, 22 of the 76 valid LLM examples have the same text as a valid example on another source. With those removed from both methods, the pooled conditional difference is -0.013 (95% CI [-0.112, 0.077]; Table 7, row "Unique-text examples only"). An independent seed per source would remove this dependence.
- LLM sampling is seeded, but sampled text can differ across GPU types and library versions. `results/main_llm_v3.jsonl` holds the rewrites behind every reported number, so the transfer tables and analyses can be recomputed from it without calling the LLM.
- Models and data are not pinned to a revision, and package versions were not recorded.
- The clean accuracies, judge agreement, NLI numbers and hand-check counts in the paper come from the notebook, not from the script.
- A fresh full run is expected to give similar, not identical, LLM numbers. I did not check how deterministic TextFooler is across TextAttack versions.

## Fresh-machine check

- [ ] `reproduce_tables.py` on a machine without my files: `[TODO: date, OS, Python, pandas, numpy, result]`
- [ ] Notebook in a fresh Kaggle session with `mode = 'pilot'` (installs and runs through Section 5.2): `[TODO: date, result, problems]`

Run the pilot check in a session that does not contain the real results folder, because Section 5.3 writes the hand-check files without a mode prefix.
