# Diabetes Data Analysis
## About project
This project demonstrates data cleaning, feature engineering, visualization, and basic machine learning analysis using a diabetes medical dataset.
## Dataset
Source: [Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/dataset/34/diabetes)
Publisher: UCI Machine Learning Repository
Description: Medical data on Pima Indian women used to predict diabetes based on health measurements
License: Public Domain
## Project structure
•	analysis.py
•	project.ipynb
•	project_clean.ipynb
•	project.csv
•	project_clean.csv
## Project Goals
•	Data cleaning
•	Identifying factors associated with diabetes
•	Finding relationships between health indicators
•	Building visualizations (histograms, boxplots, heatmaps)
•	Correlation analysis
•	Machine learning analysis
## Key Findings
# Correlation Heatmap BMI (r = 0.26) and Glucose (r = 0.19) show the strongest correlation with diabetes outcome. All other variables have near-zero correlations and are not significant predictors in this dataset.
# Distribution Plots Glucose is the clearest separator between groups — diabetic patients peak at significantly higher values (~140-150 vs ~80-90). BMI also differs noticeably, with diabetic patients showing higher values (~35-40 vs ~25-27). Age reveals a weak tendency — diabetes is more common after age 50, but the curves overlap heavily. Blood pressure shows almost no separation between groups and is the weakest predictor among the four variables.
# Glucose Levels by Age Group (Boxplot) Median glucose levels are similar across all age groups (~120-125). The widest spread is observed in senior (61-80) and young (18-40) groups, suggesting high variability in glucose levels within these age categories.
# Diabetes Distribution by Age Group (Countplot) In all age groups, the number of diabetic patients exceeds non-diabetic cases. The largest gap is in the 18-40 (~180 vs ~95) and 61-80 (~165 vs ~85) groups, confirming the dataset imbalance.
# Feature Importance (RandomForest) BMI (0.19) and Glucose (0.15) are the most important features for predicting diabetes. The remaining variables — blood pressure, insulin, skin thickness, age — show similar importance (~0.11-0.12), suggesting no single dominant predictor among them. This is consistent with the correlation heatmap findings.
## Conclusions
The analysis shows that BMI (r = 0.26) and Glucose (r = 0.19) have the strongest association with diabetes — confirmed by both the correlation heatmap and the RandomForest model. Other variables show no dominant predictive power and have similar low importance scores. Age shows a weak tendency — diabetes is more common after 50, but differences between age groups are minor. In all age groups, diabetic cases significantly outnumber non-diabetic ones, indicating a dataset imbalance that limits the generalizability of conclusions.
## Limitations:
•	The dataset is imbalanced — 506 diabetic cases vs 262 non-diabetic (~2:1), which may bias model predictions toward the diabetic group 
•	The dataset includes only Pima Indian women — results cannot be generalized to other ethnicities or males 
•	Zero values in key columns (Glucose, BMI, Insulin) were replaced with median — this may distort the real distribution 
•	BMI does not account for muscle mass or body composition 
•	No data on diet, physical activity, or medication use 
•	RandomForest was used without hyperparameter tuning — model performance could be improved
## Next Steps
•	Train and evaluate the RandomForest model properly using train/test split and accuracy metrics
•	Address dataset imbalance using oversampling techniques (e.g. SMOTE)
•	Add statistical testing (t-test) to verify differences between diabetic and non-diabetic groups
•	Try other classification models (Logistic Regression, XGBoost) and compare performance
## How to Run
•	Clone the repository
•	Install dependencies:
•	pip install pandas seaborn matplotlib scikit-learn
•	Open project.ipynb in Jupyter Notebook and run all cells

## Technologies
![Python](https://img.shields.io/badge/Python-blue)
![pandas](https://img.shields.io/badge/pandas-lightgrey)
![seaborn](https://img.shields.io/badge/seaborn-teal)
![matplotlib](https://img.shields.io/badge/matplotlib-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## Author
[vaslinx] · [GitHub]( https://github.com/vaslinx)
