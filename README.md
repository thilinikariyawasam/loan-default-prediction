# loan-default-prediction
Machine Learning project for predicting loan default risk using Python

This project applies supervised machine learning techniques to predict the risk of loan default using real-world financial data. It explores the balance between predictive accuracy and model interpretability, which is critical for practical credit-risk management in the banking industry.

**Overview**

Loan defaults directly affect banks’ profitability and stability. Traditional credit scoring methods such as Logistic Regression are interpretable but often fail to capture nonlinear borrower-behavior patterns.
In this project, multiple machine learning models were trained, tuned, and compared to determine which achieves the best predictive performance for identifying borrowers likely to default.

**Dataset**

Source: Kaggle – Loan Default Dataset

Records: 49,999 borrower-level entries

Features: Financial and demographic variables such as loan amount, credit score, interest rate, and income

Target Variable: Status (1 = Default, 0 = Non-Default)

Data Preprocessing

Missing-value imputation (median for numeric, mode for categorical)

One-Hot Encoding for categorical features

Standardization for scale-sensitive models

Class balancing using SMOTE (Synthetic Minority Oversampling Technique)

Leakage and integrity checks (duplicate, perfect-predictor, shuffled-target tests)

**Machine Learning Models**

Category	      Algorithm           	Notes
Probabilistic	  Naïve Bayes	          Simple baseline model
Statistical	    Logistic Regression	  Regulatory benchmark, interpretable
Tree-based	    Decision Tree	        Fully interpretable, but prone to overfitting
Ensemble	      Random Forest	        Strong performance, moderate interpretability
Kernel-based	  Support Vector        Effective for nonlinear boundaries
                Machine (SVM)
Neural Network	ANN (MLP)	            High capacity, requires tuning
Boosting	      AdaBoost	            Sequential weak-learner boosting
Boosting	       XGBoost	            Gradient boosting with regularization

**Model Validation**

To ensure reliability and avoid overfitting:

Cross-validation and learning-curve analysis

Shuffled-target and shallow-tree tests for data leakage

Permutation importance for feature relevance

**Results Summary**

Model	                Accuracy	  Notes
Logistic Regression	  ~83%	      Stable, interpretable
Naïve Bayes	          ~87%	      Fast, weaker on minority class
SVM	                  ~94%	      Strong nonlinear separation
ANN / Random Forest 	~100%	      Extremely high accuracy, possible overfitting; requires / XGBoost / AdaBoost                further validation

**Key Insight:**
While advanced models achieved near-perfect accuracy, realistic applications should prefer interpretable and robust approaches (Logistic Regression, SVM) validated on new data.

**Key Learnings**

Importance of data preprocessing and balancing

Trade-off between accuracy and interpretability

Necessity of validation techniques to avoid spurious overfitting

Strong predictive power of ensemble and neural models in finance

**Tools & Libraries**

Python

pandas, numpy, matplotlib, seaborn

scikit-learn

imbalanced-learn

xgboost

tensorflow / keras

**Future Improvements**

Test on full dataset (1.48M records)

Apply Explainable AI (XAI) tools (e.g., SHAP, LIME) for interpretability

Include macroeconomic variables for contextual realism

Time-series validation for temporal generalization
