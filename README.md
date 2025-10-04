# Portugal_Bank_Marketing_Campaign_Analysis
This project analyzes the Portugal Bank Marketing dataset, which contains data on direct marketing campaigns conducted by a Portuguese bank. The main objective is to predict whether a client will subscribe to a term deposit (y variable).

The dataset is highly imbalanced, with the majority of clients not subscribing. Therefore, special techniques such as SMOTE and proper model evaluation metrics were applied.

Key Steps in Analysis

Exploratory Data Analysis (EDA)
Examined categorical and numerical variables.
Found that duration strongly correlates with the target variable but leads to data leakage, so it was handled carefully.

Identified useful predictors such as:

job, marital, education, housing, loan, and poutcome.
Missing values represented as "unknown" were treated as separate categories.

Data Preprocessing

Encoding: Label Encoding and One-Hot Encoding applied to categorical variables.
Scaling: Standardization and MinMax scaling applied to numerical data.
Handling Imbalance: Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
Feature Importance

Random Forest revealed key predictors:

euribor3m, nr.employed, emp.var.rate, and poutcome.
Model Building & Evaluation

Built and compared three machine learning models:

Logistic Regression
Decision Tree
Random Forest

Metrics used:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Results
Random Forest performed the best overall, with the highest recall, F1-score, and ROC-AUC, making it the most effective for predicting client subscription.

Project Structure

Portugal_Bank_Marketing_Campaign_Analysis.ipynb → Jupyter notebook containing full analysis.
bank.csv → Dataset used (from UCI Machine Learning Repository).

Insights

Duration is a leakage variable and cannot be used directly in predictions.
Economic indicators like euribor3m and nr.employed play a significant role.
Handling class imbalance is crucial; otherwise, models become biased towards "No subscription."
Random Forest is the most reliable model for this dataset.

Technologies Used

Python (NumPy, Pandas, Matplotlib, Seaborn)
Scikit-learn (LabelEncoder, SMOTE, Logistic Regression, Decision Tree, Random Forest)

Conclusion

The project demonstrates the full machine learning pipeline: data cleaning, EDA, preprocessing, feature engineering, model training, and evaluation.
The Random Forest model proved to be the best choice for predicting whether a client subscribes to a term deposit.
