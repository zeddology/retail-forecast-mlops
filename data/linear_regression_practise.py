import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)
n_samples = 200

#Generate 3 features(independent Variables)

X1 = np.random.rand(n_samples) * 10 #0 to 10
X2 = np.random.rand(n_samples) * 5
X3 = np.random.rand(n_samples) * 8

y_true = 2 * X1 + 3 * X2 - 1 * X3 + 5
noise = np.random.normal(0, 3, n_samples) #(initialize_mean, std dev, sample_size)

y = y_true + noise

X = np.column_stack([X1, X2, X3])

print("=" * 70)
print("PRACTICE: LINEAR REGRESSION")
print("=" * 70)
print(f"\nData created:")
print(f"  - Samples: {n_samples}")
print(f"  - Features: 3 (X1, X2, X3)")
print(f"  - True relationship: y = 2*X1 + 3*X2 - 1*X3 + 5 + noise")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

#Train and Split the data following the standard 80:20 splitting
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain/Test split (80/20):")
print(f"  - Train: {len(X_train)} samples")
print(f"  - Test:  {len(X_test)} samples")

#Train LR
print(f"\n Training Linear Regression..")
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\n model trained!")

#X = 160×4 matrix:  [1,    X1[1], X2[1], X3[1]]
#                   [1,    X1[2], X2[2], X3[2]]
#                   [...   ...    ...    ...]
 #                  [1,    X1[160], X2[160], X3[160]]
#
#y = 160×1 vector:  [y[1], y[2], ..., y[160]]

#Solution (OLS formula):
#b = (X^T * X)^(-1) * X^T * y

#Where:
# X^T       = transpose of X
#  (...)^(-1) = matrix inverse
#  b         = [b0, b1, b2, b3] 
#X = 160×4 matrix:  [1,    X1[1], X2[1], X3[1]]
            
#When you call model.fit(X_train, y_train), scikit-learn:
#Takes your X_train (160×3) and y_train (160×1)
#Constructs the full X matrix with the intercept column: (160×4)
#Applies the OLS formula: b = (X^T * X)^(-1) * X^T * y
#Stores b0, b1, b2, b3 in model.coef_ and model.

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
print(f"\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nMetrics:")
print(f"  R² (train):     {r2_train:.4f}  ← explains {r2_train*100:.1f}% of variance")
print(f"  R² (test):      {r2_test:.4f}")
print(f"  RMSE (test):    {rmse_test:.4f}")
print(f"  MAE (test):     {mae_test:.4f}")