# Employee Attrition and Future Salary Prediction

## Overview
This project predicts:
- Whether an employee is likely to leave(Classification)
- What their future salary would be(Reggresion)
- The total expected financial loss due to potential employee attrition

The dataset used : **IBM HR Analytics Employee Attrition Dataset**.


## Project Workflow
1. **Attrition Prediction (Classification)**
- Models:
  - Logistic Regression (best)
  - Decision Tree
  - SVM
- Metrics:
  - F1 Score
  - ROC-AUC Score
  - Precision-Recall AUC
- Visuals:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve

2. **Simulating Future Salaries (Data Augmentation)**
-Simulates next year's salary based on PerformanceRating:
   - Rules:
      - 10% raise for high performers (`PerformanceRating = 4`)
      - 5% raise for others

3. **Future Salary Prediction (Regression)**
- Models:
  - Random Forest
  - Ridge
  - Lasso ✅ (best)
  - SVR
- Metrics:
  - R² Score
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
- Visuals:
  - Bar plots comparing R², RMSE, and MAPE
  - Scatter plot: Actual vs Predicted Future Salary

4. **Identify Likely to Stay Employees**
   - Employees with probability of staying more than 0.6 are considered likely to stay.
   - Future salary prediction focuses on these employees for realistic salary planning.

5. **Expected Salary Loss Calculation**
   - For each employee:
     ```
     Expected Loss = P(Attrition) × Future Salary
     ```
   - Aggregates total expected salary loss across the company.

## Results

- **Classification Accuracy**: (ROC-AUC Score reported)
- **Regression Accuracy**: (R² Score and RMSE reported)
- **Top Employees by Expected Salary Loss** identified.
- **Total Financial Risk** to the company estimated.