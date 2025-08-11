# Heart Disease Prediction using Decision Trees and Random Forests

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-brightgreen)

## 📌 Project Overview
This project demonstrates the implementation of Decision Trees and Random Forests for predicting heart disease using the UCI Heart Disease Dataset. The analysis includes model training, visualization, overfitting analysis, feature importance interpretation, and cross-validation.

## 📂 Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **File**: `heart.csv` (included in repository)
- **Features**: 13 clinical features including age, sex, chest pain type, etc.
- **Target**: Presence of heart disease (0 = no, 1 = yes)

## 🛠️ Implementation
### Key Steps:
1. **Data Preprocessing**
   - Load and explore the dataset
   - Split into training (70%) and testing (30%) sets

2. **Decision Tree**
   - Train with `max_depth=3` to prevent overfitting
   - Visualize the tree structure
   - Evaluate performance metrics

3. **Overfitting Analysis**
   - Compare training vs. test accuracy across different tree depths
   - Identify optimal tree depth

4. **Random Forest**
   - Train with 100 estimators
   - Compare accuracy with single decision tree
   - Analyze feature importance

5. **Model Evaluation**
   - Classification reports
   - Confusion matrices
   - 5-fold cross-validation

## 📊 Results
### Performance Metrics:
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Decision Tree       | 0.82     | 0.83      | 0.83   | 0.82     |
| Random Forest       | 0.89     | 0.89      | 0.89   | 0.89     |

### Top 5 Important Features:
1. Chest pain type (cp)
2. Thalassemia (thal)
3. Number of major vessels (ca)
4. ST depression (oldpeak)
5. Maximum heart rate (thalach)

## 📦 Files in Repository
- `heart_analysis.py`: Main Python script
- `heart.csv`: Dataset
- `decision_tree.png`: Decision tree visualization
- `overfitting_analysis.png`: Accuracy vs. tree depth plot
- `feature_importance.png`: Feature importance bar chart

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/saku2603/Task-5/tree/main
