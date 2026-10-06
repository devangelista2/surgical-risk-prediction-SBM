# SBM Stratify Training Studio

Trains risk models from a spreadsheet of patients, on your own computer, and packs the best ones for the NeuroSurg-Predict website. The first part is for clinicians; the second is for developers.

---

# For clinicians

Your spreadsheet, runs and models stay on this machine. The studio opens in your browser, but your computer serves it.

## Set up once

You need `git` and `uv`; if your Mac lacks them, ask whoever set up the studio. Then, in the Terminal app:

```bash
git clone https://github.com/devangelista2/surgical-risk-prediction-SBM.git
```

## Start

```bash
cd surgical-risk-prediction-SBM
git pull                  # get the latest version
uv run admin_panel.py
```

When Terminal shows a line starting `Running on`, open <http://localhost:5000>. The first start takes longer. Closing Terminal stops the studio.

## Train

On the **Launchpad** tab:

1. **Upload file…**: an Excel or CSV file, one row per patient.
2. Pick the outcomes to predict. Only yes/no columns appear; each gets its own models.
3. Name the run.
4. In the feature builder, click columns to add or remove them as inputs. **All Usable Columns** takes every column except dates, ID columns and the outcomes. Remove the other outcomes and anything recorded after surgery.
5. Click **Train Selected Configuration**.

If the data would make training fail, the studio stops first and says why, for example text in a number column or an outcome with fewer than 10 patients in one group.

You can close the browser tab during a run; reopening the page shows it. **Stop Training** halts it and keeps the outcomes already done. If the studio itself closed mid-run, start the run again with the same settings: finished tuning is reused.

## Read the results

On the **Results Explorer** tab, pick a run. Each outcome shows a comparison of all models, the inputs each model relies on most (the weakest are candidates to drop next time), the ROC and precision-recall curves, and a tab per model.

Scores come from test patients the model never saw:

- **AUROC**: how well the model ranks patients with the outcome above those without. 0.5 is a coin toss, 1.0 perfect.
- **Average precision**: like AUROC, but fairer when the outcome is rare.
- **Recall**: sensitivity. **Precision**: positive predictive value. **Specificity** and **NPV** as usual.

Charts:

- **Confusion matrix**: real outcome against the model's call. Off the diagonal are missed cases and false alarms.
- **ROC curve**: better the closer it bends to the top-left.
- **Precision-recall curve**: prefer it over ROC for rare outcomes. Higher is better.
- **Feature importance**: how much the score drops when one input is scrambled.

Per-model charts and combined curves are saved as PNG and PDF under `outputs/studio_runs/`.

## Send models to NeuroSurg-Predict

At the bottom of a run, pick one model per outcome (the best AUROC is preselected; **Leave out** skips one), name the freeze and click **Freeze Selected Models**. Then click **Download zip** and send the zip to the website's maintainer.

A warning that a model "cannot load" means it uses a date or multi-label column. Remove that column and train again.

## Advanced training settings

The defaults are fine to leave.

- **Learners**: gradient boosting (`hgb`), random forest (`rf`), logistic regression (`lr`), support vector machine (`svc`).
- **Split strategy**: **Temporal** (default) tests on the most recent patients and needs the **Date column**. **Random** holds back a random share. **Predefined** reads `train` or `test` from the **Split column**.
- **Test size**: share of patients held back for testing (0.20 is one in five).
- **Threshold val**: share of training patients used to choose the cut-off.
- **Min recall**: the cut-off must catch at least this share of patients with the outcome; among those, the studio takes the most precise. If none reaches it, it takes the one that catches the most.
- **F-beta, FN cost, FP cost**: tie-breakers for the cut-off.
- **Grid search intensity**: Light is fastest, Heavy most thorough.
- **Re-tune even when saved tuning matches**: redo tuning instead of reusing it.
- **Load settings from a previous run**: copies every setting from an earlier run.
- **Advanced: column types**: fix how a column is read, for example codes read as numbers.

---

# For developers

```text
.
├── admin_panel.py            # Web app: upload, tune, train, results, freeze
├── templates/index.html      # Front-end
├── static/                   # Favicons, theme picker
├── src/
│   ├── train.py              # Training, spawned per target
│   ├── tune.py               # Grid-search tuning
│   ├── preprocessing.py      # Date and multi-label transformers
│   └── utils/                # Logging and plotting
├── configs/                  # grid_search_{light,medium,heavy}.json
├── data/                     # Uploaded datasets (git-ignored)
└── outputs/
    ├── studio_runs/          # One folder per run (new runs git-ignored)
    ├── tuning_cache/         # Saved tuning (git-ignored)
    └── freezes/              # Models packed for neurosurg-predict (git-ignored)
```

Each model folder in a run holds `pipeline.joblib`, `metrics.json`, `decision_policy.json`, `test_predictions.csv` and the plots (300 DPI PNG plus PDF). Pipelines with date or multi-label columns need `src/` on the import path.

```python
import joblib
import pandas as pd

pipeline = joblib.load("outputs/studio_runs/<run>/<target>/<model>/pipeline.joblib")
new_patients = pd.read_csv("new_patients.csv")  # same feature columns as <target>/metadata.json
probabilities = pipeline.predict_proba(new_patients)[:, 1]
```

Cut at `threshold` from `decision_policy.json`, not with `pipeline.predict`, which cuts at 0.5.
