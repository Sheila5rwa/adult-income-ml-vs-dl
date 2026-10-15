# Predicting Economic Opportunity from Education and Work Characteristics
**Traditional Machine Learning vs. Deep Learning on the UCI Adult (Census Income) dataset**

Machine Learning module – Summative project

| Resource | Link |
|---|---|
| Report (PDF) | submitted on the course portal |
| Google Drive (raw dataset + saved models) | *add link* |
| Video presentation | *add link* |

## Problem and mission
This project supports the mission of **Ascend Rwanda**: widening access to literacy, language, digital and financial skills for people whose background limits their economic opportunity. It asks which personal, educational and work characteristics are most strongly associated with reaching a higher income band, and how well classical ML and deep learning models can predict that outcome. The task is binary classification: annual income `>50K` vs `<=50K`.

## Dataset
* **Source:** UCI Machine Learning Repository, *Adult* (Becker & Kohavi, 1996), extracted from the 1994 US Census database.
* **Size:** 48,842 records (32,561 in `adult.data` and 16,281 in `adult.test`), 14 attributes, binary target, about 24% positive.
* **Not included in this repository.** Download `adult.zip` from the Google Drive folder above (or from UCI) and place it in the project root or in `data/`. The notebook also downloads it from UCI automatically if no local copy is found.

## Results (held-out test set)
| Model | ROC-AUC | PR-AUC | F1 (tuned threshold) |
|---|---|---|---|
| E7 – Tuned Histogram Gradient Boosting (best ML) | **0.926** | **0.824** | **0.719** |
| E11 – Keras Functional API with embeddings (best DL) | 0.912 | 0.781 | 0.689 |

13 experiments in total (7 scikit-learn, 6 Keras). The full table is in `outputs/experiment_table.csv`, and every run is logged to TensorBoard in `logs/`.

## Repository structure
```
├── Adult_Income_ML_vs_DL.ipynb   # full pipeline, executed with outputs
├── requirements.txt
├── logs/                         # TensorBoard logs for all 13 experiments (E1–E13)
└── outputs/
    ├── experiment_table.csv      # validation metrics + hyperparameters per run
    ├── final_test_results.csv    # test metrics for best ML and DL models
    ├── subgroup_errors.csv       # error analysis by education, sex, race
    └── figures/                  # all figures used in the report
```
Saved models (`best_traditional_ml_E7.joblib`, `best_deep_learning_E11.keras` / `.h5`, `dl_preprocessing.joblib`) are in the Google Drive folder.

## How to reproduce
**Google Colab**
1. Open the notebook in Colab (File → Open notebook → GitHub → paste this repo URL).
2. Upload `adult.zip` to the session (or let the notebook download it).
3. Run the cell `!pip install -q -r requirements.txt` if needed (clone the repo first), then *Runtime → Run all*.

**Locally**
```bash
git clone <this-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Adult_Income_ML_vs_DL.ipynb
```
A full run takes about 10–15 minutes on a CPU. All randomness is seeded (`SEED = 42`).

**View the experiment logs**
```bash
tensorboard --logdir logs
```

## Pipeline summary
1. **Cleaning:** strip the trailing `.` in test labels; keep `?` as an explicit `Unknown` category; remove 24 duplicate training rows; drop `fnlwgt` (sampling weight) and `education` (duplicate of `education_num`).
2. **Feature engineering:** log and indicator transforms for capital gain and loss, `age²`, married flag, five education tiers, hours bands, US-born flag.
3. **Splits:** stratified 80/20 train/validation from `adult.data`; official `adult.test` held out for the final evaluation only.
4. **Traditional ML:** Logistic Regression (base, engineered, class-weighted), Random Forest (default, tuned), Histogram Gradient Boosting (default, tuned) with `RandomizedSearchCV`.
5. **Deep learning:** `tf.data` pipelines; Sequential MLPs (baseline, high learning rate, deep + dropout/BN/L2); Functional API embedding networks (baseline, class weights, LR schedule); early stopping on validation AUC.
6. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC; validation-tuned thresholds; confusion matrices; ROC/PR curves; learning curves; subgroup error analysis; permutation importance.

## Acknowledgements
Dataset: Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
