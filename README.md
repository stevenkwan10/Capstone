# D606 Capstone — Random Forest Analysis

Overview
- This notebook trains Random Forest models to predict Airbnb booking (`is_booked`) using baseline and enriched feature sets, performs GroupKFold cross-validation, and compares AUC performance on a holdout test set.

Notebooks
- Main notebook: [working_folder/D606_Task2_random_forest.ipynb](working_folder/D606_Task2_random_forest.ipynb)

How to run
- Recommended: open the notebook in VS Code or Jupyter and run cells in order.
- Install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Suggested minimal requirements
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- category_encoders
- joblib

Reproducibility
- Random seeds are set in the notebook (`random.seed(42)`, `np.random.seed(42)`). Ensure `PYTHONHASHSEED=42` if re-running.
