# SBM Stratify Training Studio

A streamlined, robust, and highly configurable machine learning studio for medical tabular data. It handles both classification and regression automatically, supports temporal/predefined/random splitting strategies, tunes and trains Scikit-Learn models, and exports publication-ready scientific plots — all from a single web app.

## 📂 Project Structure

```text
.
├── admin_panel.py            # The web app: upload data, tune, train, browse results, freeze models
├── templates/index.html      # Studio front-end
├── static/                   # Logo and assets
├── src/
│   ├── train.py              # Training script (spawned by the studio per target)
│   ├── tune.py               # Grid-search tuning script
│   ├── preprocessing.py      # Custom scikit-learn transformers (dates, multilabel)
│   └── utils/                # Logging and publication-ready plotting utilities
├── configs/
│   ├── grid_search_light.json    # Small grid — fastest tuning
│   ├── grid_search_medium.json   # Balanced grid (default)
│   └── grid_search_heavy.json    # Large grid — most thorough tuning
├── data/                     # Datasets (.xlsx / .xls / .csv)
└── outputs/                  # Studio runs, tuning cache, and frozen models
```

## 🖥️ Running the studio

```bash
python admin_panel.py      # http://localhost:5000/
```

From the browser you can upload a dataset, pick the outcomes, features and learners, adjust the
advanced training settings, and launch training. Results are written under
`outputs/studio_runs/` and can be browsed in the Results Explorer or frozen for deployment.

A training run keeps going when the browser tab closes; reopening the page shows it. If the studio
process is closed mid-run, the page says so on the next start, and starting the run again with
the same settings skips the tuning already done.

### Stopping a run

While a run is going, the **Train Selected Configuration** button turns red and becomes a
**Stop Training** button. Clicking it halts the current tuning/training subprocess and cancels
the remaining targets. Any target that already finished is kept, and its tuning is cached —
restarting with the same settings resumes from there.

### Grid search intensity

Under **Advanced training settings → Grid search intensity** you can choose how many
hyper-parameter combinations tuning explores:

- **Light** — fewer values, fastest, good for a quick pass.
- **Medium** — balanced default.
- **Heavy** — more values, slower but more precise.

Picking a preset fills the *Search grid (JSON)* box, which you can still edit by hand to override it.

## 🧠 Inference (Loading Saved Weights)

The script automatically saves the entire trained pipeline (imputers, scalers, encoders, and the model itself) as `pipeline.joblib`. 
To use this model on new, unseen patients later:

```python
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load("benchmark_output/rf/pipeline.joblib")

# Load new patient data (must contain the same features defined in data_config.json)
new_patients = pd.read_csv("new_patients.csv")

# Predict directly! The pipeline handles all preprocessing internally.
predictions = pipeline.predict(new_patients)
probabilities = pipeline.predict_proba(new_patients)
```

---

## 📊 How to Read the Generated Plots

When training finishes, the output folder will contain a `metrics.json` file and several high-resolution (`300 DPI`) plots tailored for scientific publication.

### 1. Confusion Matrix (`confusion_matrix.png`)
* **What it shows:** A grid comparing the *Actual* patient outcomes (True Label) against the *Predicted* outcomes by the model.
* **How to read it:** * **Diagonal cells** (top-left to bottom-right) represent correct predictions (True Positives and True Negatives). 
  * **Off-diagonal cells** represent errors (False Positives and False Negatives). In clinical settings, predicting a complication when there isn't one (False Positive) is usually preferred over missing a fatal complication (False Negative).

### 2. ROC Curve (`roc_curve.png`)
* **What it shows:** The trade-off between the True Positive Rate (Sensitivity) and the False Positive Rate (1 - Specificity) across different probability thresholds.
* **How to read it:** * The dashed diagonal line represents random guessing (AUC = 0.50).
  * The closer the solid curve gets to the top-left corner, the better the model is at distinguishing between the two classes. 
  * **AUC (Area Under the Curve):** A value of 1.0 means perfect separation. A value > 0.80 is generally considered excellent for clinical models.

### 3. Precision-Recall Curve (`pr_curve.png`)
* **What it shows:** The trade-off between Precision (Positive Predictive Value) and Recall (Sensitivity). 
* **How to read it:** This plot is highly recommended over the ROC curve when your dataset is **imbalanced** (e.g., only 5% of patients have the complication). A model that stays close to the top-right corner is highly effective at finding the rare minority class without throwing too many false alarms.

### 4. Actual vs Predicted Plot (`actual_vs_predicted.png`)
* **What it shows:** Used *only* for regression tasks (e.g., predicting "Days of hospitalization"). It plots the model's prediction on the Y-axis against the actual truth on the X-axis.
* **How to read it:** * The red dashed line represents perfect prediction ($y = x$). 
  * Points clustered tightly along this line indicate high accuracy. 
  * If points fan out heavily at higher values, the model is struggling to predict extreme/high outcomes.
