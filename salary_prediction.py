# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import numpy as np

#Load the Dataset
df = pd.read_csv('SalaryData.csv')

#Encode Categorical Features
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

#Define Features (X) and Target (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

#Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (Attrition = Yes)

#Evaluate the Model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))

#Plot the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba)))
plt.plot([0,1], [0,1], color='red', linestyle='--')  # Random guessing line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

# Increment column and future salary column created(PHASE 2)
df["Increment"] = df["PerformanceRating"].apply(lambda x: 1.10 if x == 4 else 1.05)
df["FutureSalary"] = df["MonthlyIncome"] * df["Increment"]


# (PHASE 3)
# Define Features (X) and Target (y) for Salary Prediction
X_salary = df.drop(['Attrition', 'FutureSalary', 'Increment'], axis=1)
y_salary = df['FutureSalary']

# Split into Train and Test for Salary Prediction
X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

# Train Random Forest Regressor
salary_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
salary_regressor.fit(X_train_sal, y_train_sal)

# Predict Future Salaries
y_pred_sal = salary_regressor.predict(X_test_sal)

# Evaluate Salary Prediction Model
r2 = r2_score(y_test_sal, y_pred_sal)
rmse = np.sqrt(mean_squared_error(y_test_sal, y_pred_sal))

print("\nRegression Model Evaluation for Future Salary Prediction:")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Scatter plot: Actual vs Predicted Future Salaries
plt.figure(figsize=(8,6))
plt.scatter(y_test_sal, y_pred_sal, color='blue', alpha=0.6)
plt.plot([y_test_sal.min(), y_test_sal.max()], [y_test_sal.min(), y_test_sal.max()], color='red', linestyle='--')
plt.xlabel('Actual Future Salary')
plt.ylabel('Predicted Future Salary')
plt.title('Actual vs Predicted Future Salary')
plt.grid(True)
plt.show()