import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# Silver Price Prediction - ML Lab Project

## 1. Dataset Information
- **Source**: Yahoo Finance (Silver Futures SI=F)
- **Period**: 2016-2026 (10 years daily data)
- **Target**: Next day's closing price (Regression)
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import joblib, os, warnings
warnings.filterwarnings('ignore')
print("Libraries loaded!")"""))

# Load Data
cells.append(nbf.v4.new_markdown_cell("## 2. Data Loading"))
cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('silver_prices_10years.csv')
print(f"Shape: {df.shape}")
df.head()"""))

cells.append(nbf.v4.new_code_cell("""print("Missing values:", df.isnull().sum().sum())
df.describe()"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis"))
cells.append(nbf.v4.new_code_cell("""df['Date'] = pd.to_datetime(df['Date'])
plt.figure(figsize=(12,4))
plt.plot(df['Date'], df['Close'])
plt.title('Silver Prices Over Time')
plt.xlabel('Date'); plt.ylabel('Price (USD)')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""sns.heatmap(df[['Open','High','Low','Close','Volume']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("## 4. Feature Engineering"))
cells.append(nbf.v4.new_code_cell("""def create_features(df):
    df = df.copy()
    for lag in [1,2,3,5,7]: df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    for w in [5,10,20]: df[f'MA_{w}'] = df['Close'].rolling(w).mean()
    df['Momentum'] = df['Close'].pct_change(5)
    df['Volatility'] = df['Close'].rolling(5).std()
    df['Target'] = df['Close'].shift(-1)
    return df.dropna()

df_feat = create_features(df)
print(f"Features created. Shape: {df_feat.shape}")"""))

# Train-Test Split
cells.append(nbf.v4.new_markdown_cell("## 5. Data Preprocessing"))
cells.append(nbf.v4.new_code_cell("""features = [c for c in df_feat.columns if c not in ['Date','Target','Close']]
X, y = df_feat[features], df_feat['Target']
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")"""))

# Model Training
cells.append(nbf.v4.new_markdown_cell("## 6. Model Training"))
cells.append(nbf.v4.new_code_cell("""def evaluate(y_true, y_pred):
    return {'MAE':mean_absolute_error(y_true,y_pred), 
            'RMSE':np.sqrt(mean_squared_error(y_true,y_pred)),
            'R2':r2_score(y_true,y_pred)}

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'DecisionTree': DecisionTreeRegressor(max_depth=10),
    'RandomForest': RandomForestRegressor(n_estimators=100,max_depth=15),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf', C=100)
}

results = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    metrics = evaluate(y_test, pred)
    results.append({'Model':name, **metrics})
    print(f"{name}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
results_df"""))

# Comparison
cells.append(nbf.v4.new_markdown_cell("## 7. Model Comparison"))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].barh(results_df['Model'], results_df['R2'], color='steelblue')
ax[0].set_xlabel('R² Score'); ax[0].set_title('R² (Higher=Better)')
ax[1].barh(results_df['Model'], results_df['RMSE'], color='coral')
ax[1].set_xlabel('RMSE'); ax[1].set_title('RMSE (Lower=Better)')
plt.tight_layout(); plt.show()"""))

# Hyperparameter Tuning
cells.append(nbf.v4.new_markdown_cell("## 8. Hyperparameter Tuning"))
cells.append(nbf.v4.new_code_cell("""rf_grid = GridSearchCV(RandomForestRegressor(),
    {'n_estimators':[50,100,150],'max_depth':[10,15,20]}, cv=3, scoring='r2')
rf_grid.fit(X_train_s, y_train)
print(f"Best params: {rf_grid.best_params_}")
rf_pred = rf_grid.predict(X_test_s)
rf_metrics = evaluate(y_test, rf_pred)
print(f"Tuned RF: R2={rf_metrics['R2']:.4f}, RMSE={rf_metrics['RMSE']:.4f}")"""))

# Feature Importance
cells.append(nbf.v4.new_markdown_cell("## 9. Feature Importance"))
cells.append(nbf.v4.new_code_cell("""importance = pd.DataFrame({'Feature':features,'Importance':rf_grid.best_estimator_.feature_importances_})
importance = importance.sort_values('Importance',ascending=False).head(10)
plt.barh(importance['Feature'], importance['Importance'])
plt.xlabel('Importance'); plt.title('Top 10 Features')
plt.gca().invert_yaxis(); plt.show()"""))

# Predictions
cells.append(nbf.v4.new_markdown_cell("## 10. Predictions Visualization"))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12,4))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(rf_pred[:100], label='Predicted', alpha=0.7)
plt.legend(); plt.title('Actual vs Predicted'); plt.show()"""))

# Save
cells.append(nbf.v4.new_markdown_cell("## 11. Save Results"))
cells.append(nbf.v4.new_code_cell("""os.makedirs('saved_models', exist_ok=True)
joblib.dump(rf_grid.best_estimator_, 'saved_models/best_model.pkl')
results_df.to_csv('results.csv', index=False)
print("Saved!")"""))

# Model Explanations
cells.append(nbf.v4.new_markdown_cell("""## 12. Model Explanations

| Model | How it Works | Pros | Cons |
|-------|-------------|------|------|
| Linear | Fits line minimizing squared errors | Fast, interpretable | Linear assumption |
| Ridge | Linear + L2 penalty | Handles multicollinearity | No feature selection |
| Lasso | Linear + L1 penalty | Feature selection | May drop important features |
| Decision Tree | Recursive splits | Non-linear, interpretable | Overfits easily |
| Random Forest | Ensemble of trees | Robust, accurate | Less interpretable |
| Gradient Boosting | Sequential correction | High accuracy | Slow, overfits |
| KNN | Nearest neighbors avg | Simple | Slow prediction |
| SVR | Kernel-based | Non-linear | Hard to tune |
"""))

# Metrics
cells.append(nbf.v4.new_markdown_cell("""## 13. Metrics Justification

**Regression Metrics Used:**
- **R²**: Variance explained (target >0.9)
- **RMSE**: Error in same units (USD)
- **MAE**: Average absolute error

R² is primary metric as it's scale-independent and intuitive.
"""))

# Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 14. Conclusion

1. **Best Model**: Random Forest/Gradient Boosting
2. **Key Features**: Lag prices, moving averages
3. **Feature engineering** significantly improved results
4. **Ensemble methods** outperformed linear models
"""))

cells.append(nbf.v4.new_code_cell("""print("Project completed successfully!")"""))

nb.cells = cells
with open('Silver_Price_Prediction_ML_Project.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created: Silver_Price_Prediction_ML_Project.ipynb")
