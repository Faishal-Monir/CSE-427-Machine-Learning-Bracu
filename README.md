# CSE-427 (Machine Learning) - Spring 2025

Final project repository for CSE-427. The main project trains multiple ML models to predict student outcomes (Dropout, Graduate, Enrolled) and compares their performance.

## Project Overview
- Task: Multiclass classification of student outcomes.
- Models: Decision Tree, Random Forest, Logistic Regression, AdaBoost, KNN, Neural Network, and an ensemble (VotingClassifier).
- Evaluation: Accuracy, classification report, and confusion matrices.
- Extras: Feature importance comparisons and batch prediction on a separate test dataset.

## Dataset
The notebook expects a training and test CSV with a `Target` column:
- `Target` labels: `Dropout`, `Graduate`, `Enrolled` (mapped to 0/1/2).
- Local copies are in `LAB/CSE-427(Final Project)/train.csv` and `LAB/CSE-427(Final Project)/test.csv`.
- The notebook also includes Google Drive links for loading the same data directly.

## How to Run
1) Open the notebook in Jupyter or Colab:
   - `LAB/CSE-427(Final Project)/CSE-427_Final project_Sec-1_Group-9 .ipynb`
2) Install dependencies (suggested):
   - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `joblib`, `xlsxwriter`
3) If running locally, replace the Google Drive links with local paths, for example:

```python
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
```

## Outputs
The notebook saves trained models and predictions in the working directory:
- Model artifacts: `decision_tree_model.pkl`, `random_forest_model.pkl`, `logistic_regression_model.pkl`, `adaboost_model.pkl`, `knn_model.pkl`, `neural_network_model.pkl`
- Predictions workbook: `model_predictions.xlsx`

## Repository Structure
- `LAB/CSE-427(Final Project)/` : Final project notebook, data, and research papers
- `Ourproject/` : Report files and supporting materials
- `LAB/LAB-1` ... `LAB/LAB-6` : Course lab submissions

## Course Info
Course: CSE-427 (Machine Learning)  
Semester: Spring 2025
