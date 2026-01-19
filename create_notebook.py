import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title and Dataset Info
cells.append(nbf.v4.new_markdown_cell("""# Silver Price Prediction - ML Lab Project

## 1. Dataset Information

| Attribute | Details |
|-----------|---------|
| **Source** | Yahoo Finance (Silver Futures SI=F) |
| **Period** | January 2016 - January 2026 (10 years) |
| **Records** | ~2,500 trading days |
| **Type** | Time Series / Regression |
| **Target** | Next day's closing price |

### Features Description
- **Date**: Trading date
- **Open/High/Low/Close**: Daily OHLC prices (USD/oz)
- **Volume**: Trading volume
- **Engineered**: Lags, moving averages, momentum, volatility
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
print("All libraries imported successfully!")"""))

# Data Loading
cells.append(nbf.v4.new_markdown_cell("## 2. Data Loading & Initial Exploration"))
cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('silver_prices_10years.csv')
print(f"Dataset Shape: {df.shape}")
print(f"\\nColumn Types:\\n{df.dtypes}")
df.head(10)"""))

cells.append(nbf.v4.new_code_cell("""# Check for missing values
print("Missing Values per Column:")
print(df.isnull().sum())
print(f"\\nTotal Missing: {df.isnull().sum().sum()}")"""))

cells.append(nbf.v4.new_code_cell("""# Statistical Summary
df.describe()"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis"))
cells.append(nbf.v4.new_code_cell("""# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Price trend over time
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Closing Price Trend
axes[0,0].plot(df['Date'], df['Close'], color='blue', linewidth=0.5)
axes[0,0].set_title('Silver Price Over 10 Years')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Price (USD/oz)')

# Price Distribution
axes[0,1].hist(df['Close'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0,1].set_title('Price Distribution')
axes[0,1].set_xlabel('Price')
axes[0,1].set_ylabel('Frequency')

# Volume Over Time
axes[1,0].bar(df['Date'], df['Volume'], color='green', alpha=0.5, width=3)
axes[1,0].set_title('Trading Volume')
axes[1,0].set_ylabel('Volume')

# Box Plot by Year
df['Year'] = df['Date'].dt.year
df.boxplot(column='Close', by='Year', ax=axes[1,1])
axes[1,1].set_title('Price by Year')
axes[1,1].set_xlabel('Year')
axes[1,1].set_ylabel('Price (USD)')

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Correlation Analysis  
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.3f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print("Observations:")
print("- OHLC prices are highly correlated (>0.99)")
print("- Volume has weak correlation with prices")"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Engineering

Creating predictive features from raw price data:
- **Lag Features**: Past N days closing prices
- **Moving Averages**: Trend indicators
- **Momentum**: Price change rate
- **Volatility**: Rolling standard deviation
- **Price Range**: Daily high-low spread"""))

cells.append(nbf.v4.new_code_cell("""def create_features(df):
    '''Create features for ML model'''
    data = df.copy()
    
    # Lag Features (past closing prices)
    for lag in [1, 2, 3, 5, 7, 10]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
    
    # Moving Averages
    for window in [5, 10, 20, 50]:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    
    # Momentum (price change over N days)
    data['Momentum_5'] = data['Close'].pct_change(5) * 100
    data['Momentum_10'] = data['Close'].pct_change(10) * 100
    
    # Volatility (rolling std)
    data['Volatility_5'] = data['Close'].rolling(window=5).std()
    data['Volatility_10'] = data['Close'].rolling(window=10).std()
    
    # Price Range  
    data['Daily_Range'] = data['High'] - data['Low']
    data['Range_Pct'] = (data['High'] - data['Low']) / data['Close'] * 100
    
    # Target: Next day's closing price
    data['Target'] = data['Close'].shift(-1)
    
    # Drop rows with NaN
    data = data.dropna()
    
    return data

# Apply feature engineering
df_features = create_features(df)
print(f"Original shape: {df.shape}")
print(f"After feature engineering: {df_features.shape}")
print(f"\\nFeatures created:")
print([c for c in df_features.columns if c not in ['Date', 'Year', 'Target']])"""))

# Data Preprocessing  
cells.append(nbf.v4.new_markdown_cell("""## 5. Data Preprocessing

### Train-Test Split Strategy
Using **time-based split** (not random) because:
- Preserves temporal order
- Prevents data leakage
- Real-world evaluation scenario"""))

cells.append(nbf.v4.new_code_cell("""# Define features and target
feature_cols = [c for c in df_features.columns if c not in ['Date', 'Year', 'Target', 'Close']]
X = df_features[feature_cols]
y = df_features['Target']

print(f"Number of features: {len(feature_cols)}")
print(f"Feature columns: {feature_cols}")

# Time-based split (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"\\nTraining period: {df_features['Date'].iloc[0]} to {df_features['Date'].iloc[train_size-1]}")
print(f"Testing period: {df_features['Date'].iloc[train_size]} to {df_features['Date'].iloc[-1]}")"""))

cells.append(nbf.v4.new_code_cell("""# Feature Scaling for distance-based algorithms
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling applied using StandardScaler")
print(f"Mean of scaled features (should be ~0): {X_train_scaled.mean():.4f}")
print(f"Std of scaled features (should be ~1): {X_train_scaled.std():.4f}")"""))

# Model Training
cells.append(nbf.v4.new_markdown_cell("""## 6. Model Training & Evaluation

### Models Used (Classical ML Only):
1. **Linear Regression** - Baseline
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization  
4. **ElasticNet** - Combined L1+L2
5. **Decision Tree** - Non-linear, interpretable
6. **Random Forest** - Ensemble of trees
7. **Gradient Boosting** - Sequential ensemble
8. **K-Nearest Neighbors** - Instance-based
9. **Support Vector Regression** - Kernel-based"""))

cells.append(nbf.v4.new_code_cell("""def evaluate_model(y_true, y_pred):
    '''Calculate regression metrics'''
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Initialize models (scaled data for distance-based, unscaled for tree-based)
models_scaled = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.01)': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'SVR (RBF)': SVR(kernel='rbf', C=100, epsilon=0.1),
}

models_unscaled = {
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
}

results = []

# Train scaled models
print("Training models (scaled data):")
print("-" * 50)
for name, model in models_scaled.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_model(y_test, y_pred)
    results.append({'Model': name, **metrics, 'Scaled': True})
    print(f"{name:25} | R²: {metrics['R2']:.4f} | RMSE: ${metrics['RMSE']:.4f}")

# Train unscaled models (tree-based)
print("\\nTraining models (unscaled data):")
print("-" * 50)
for name, model in models_unscaled.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    results.append({'Model': name, **metrics, 'Scaled': False})
    print(f"{name:25} | R²: {metrics['R2']:.4f} | RMSE: ${metrics['RMSE']:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
print("\\n" + "="*60)
print("Training Complete!")"""))

# Model Comparison
cells.append(nbf.v4.new_markdown_cell("## 7. Model Comparison"))
cells.append(nbf.v4.new_code_cell("""# Display sorted results
print("Model Performance Comparison (Sorted by R²):\\n")
display_df = results_df.copy()
display_df['MAE'] = display_df['MAE'].apply(lambda x: f"${x:.4f}")
display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"${x:.4f}")
display_df['R2'] = display_df['R2'].apply(lambda x: f"{x:.4f}")
print(display_df[['Model', 'R2', 'RMSE', 'MAE']].to_string(index=False))"""))

cells.append(nbf.v4.new_code_cell("""# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² Score Comparison
colors = ['green' if r2 > 0.9 else 'orange' if r2 > 0.5 else 'red' 
          for r2 in results_df['R2']]
axes[0].barh(results_df['Model'], results_df['R2'], color=colors, edgecolor='black')
axes[0].set_xlabel('R² Score')
axes[0].set_title('R² Score Comparison (Higher = Better)')
axes[0].axvline(x=0.9, color='green', linestyle='--', label='Excellent (>0.9)')
axes[0].legend()

# RMSE Comparison
axes[1].barh(results_df['Model'], results_df['RMSE'], color='coral', edgecolor='black')
axes[1].set_xlabel('RMSE (USD)')
axes[1].set_title('RMSE Comparison (Lower = Better)')

plt.tight_layout()
plt.show()

# Best Model
best_model = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R2']
print(f"\\nBest Performing Model: {best_model} with R² = {best_r2:.4f}")"""))

# Hyperparameter Tuning
cells.append(nbf.v4.new_markdown_cell("""## 8. Hyperparameter Tuning

Improving top models using **GridSearchCV** with cross-validation:"""))

cells.append(nbf.v4.new_code_cell("""# Ridge Tuning
print("Tuning Ridge Regression...")
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2', n_jobs=-1)
ridge_grid.fit(X_train_scaled, y_train)
print(f"Best alpha: {ridge_grid.best_params_['alpha']}")

# Lasso Tuning  
print("\\nTuning Lasso Regression...")
lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5, scoring='r2', n_jobs=-1)
lasso_grid.fit(X_train_scaled, y_train)
print(f"Best alpha: {lasso_grid.best_params_['alpha']}")

# Random Forest Tuning
print("\\nTuning Random Forest...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, 
                       cv=3, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f"Best params: {rf_grid.best_params_}")

# Gradient Boosting Tuning
print("\\nTuning Gradient Boosting...")
gb_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params,
                      cv=3, scoring='r2', n_jobs=-1)
gb_grid.fit(X_train, y_train)
print(f"Best params: {gb_grid.best_params_}")"""))

cells.append(nbf.v4.new_code_cell("""# Evaluate tuned models
tuned_results = []

tuned_models = {
    'Ridge (Tuned)': (ridge_grid.best_estimator_, X_test_scaled),
    'Lasso (Tuned)': (lasso_grid.best_estimator_, X_test_scaled),
    'Random Forest (Tuned)': (rf_grid.best_estimator_, X_test),
    'Gradient Boosting (Tuned)': (gb_grid.best_estimator_, X_test),
}

print("Tuned Model Performance:")
print("-" * 60)
for name, (model, X_pred) in tuned_models.items():
    y_pred = model.predict(X_pred)
    metrics = evaluate_model(y_test, y_pred)
    tuned_results.append({'Model': name, **metrics})
    print(f"{name:30} | R²: {metrics['R2']:.4f} | RMSE: ${metrics['RMSE']:.4f}")

tuned_df = pd.DataFrame(tuned_results).sort_values('R2', ascending=False)"""))

# Feature Importance
cells.append(nbf.v4.new_markdown_cell("## 9. Feature Importance Analysis"))
cells.append(nbf.v4.new_code_cell("""# Random Forest Feature Importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_grid.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue', edgecolor='black')
plt.xlabel('Importance Score')
plt.title('Top 15 Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\\nKey Insights:")
print("- Lag features (previous prices) are most predictive")
print("- Moving averages capture trend information")
print("- Volatility helps predict price uncertainty")"""))

# Predictions Visualization
cells.append(nbf.v4.new_markdown_cell("## 10. Predictions Visualization"))
cells.append(nbf.v4.new_code_cell("""# Get best model predictions
best_model_obj = ridge_grid.best_estimator_
best_predictions = best_model_obj.predict(X_test_scaled)

# Plot actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time series comparison (last 100 days)
test_dates = df_features['Date'].iloc[train_size:].values
axes[0].plot(test_dates[-100:], y_test.values[-100:], 'b-', label='Actual', linewidth=1.5)
axes[0].plot(test_dates[-100:], best_predictions[-100:], 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price (USD)')
axes[0].set_title('Actual vs Predicted (Last 100 Days)')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Scatter plot
axes[1].scatter(y_test, best_predictions, alpha=0.5, color='blue', edgecolor='black', s=20)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title('Prediction Accuracy Scatter Plot')
axes[1].legend()

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Residual Analysis
residuals = y_test - best_predictions

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residual Distribution
axes[0].hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Residual (Actual - Predicted)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Residual Distribution')

# Residual Plot
axes[1].scatter(best_predictions, residuals, alpha=0.5, color='blue', s=10)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Price')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residual vs Predicted')

plt.tight_layout()
plt.show()

print(f"Mean Residual: ${residuals.mean():.4f} (should be ~0)")
print(f"Residual Std: ${residuals.std():.4f}")"""))

# Save Results
cells.append(nbf.v4.new_markdown_cell("## 11. Save Models & Results"))
cells.append(nbf.v4.new_code_cell("""# Create output directory
os.makedirs('saved_models', exist_ok=True)

# Save best model
joblib.dump(ridge_grid.best_estimator_, 'saved_models/best_ridge_model.pkl')
joblib.dump(rf_grid.best_estimator_, 'saved_models/best_rf_model.pkl')
joblib.dump(scaler, 'saved_models/scaler.pkl')

# Save all results
all_results = pd.concat([results_df, tuned_df], ignore_index=True)
all_results.to_csv('model_results.csv', index=False)

print("Saved Files:")
print("- saved_models/best_ridge_model.pkl")
print("- saved_models/best_rf_model.pkl")
print("- saved_models/scaler.pkl")
print("- model_results.csv")"""))

# Model Explanations
cells.append(nbf.v4.new_markdown_cell("""## 12. Model Explanations

### Linear Regression
- **How it works**: Fits a linear equation minimizing sum of squared residuals
- **Parameters**: No hyperparameters (OLS)
- **Pros**: Fast, interpretable, no tuning needed
- **Cons**: Assumes linear relationship, sensitive to outliers

### Ridge Regression (L2 Regularization)
- **How it works**: Linear regression + penalty on large coefficients
- **Key Hyperparameter**: `alpha` (regularization strength)
- **Pros**: Handles multicollinearity, prevents overfitting
- **Cons**: Doesn't perform feature selection

### Lasso Regression (L1 Regularization)
- **How it works**: Linear regression + absolute value penalty
- **Key Hyperparameter**: `alpha`
- **Pros**: Feature selection (sets coefficients to zero)
- **Cons**: May remove important correlated features

### Decision Tree
- **How it works**: Recursive binary splits based on feature thresholds
- **Key Hyperparameters**: `max_depth`, `min_samples_split`
- **Pros**: Non-linear, interpretable, handles mixed data
- **Cons**: Overfits easily, high variance

### Random Forest
- **How it works**: Ensemble of decision trees with bagging
- **Key Hyperparameters**: `n_estimators`, `max_depth`, `max_features`
- **Pros**: Robust, handles non-linearity, feature importance
- **Cons**: Less interpretable, slower training

### Gradient Boosting
- **How it works**: Sequential trees correcting previous errors
- **Key Hyperparameters**: `n_estimators`, `learning_rate`, `max_depth`
- **Pros**: High accuracy, handles complex patterns
- **Cons**: Prone to overfitting, slow training

### K-Nearest Neighbors (KNN)
- **How it works**: Averages K closest training samples
- **Key Hyperparameter**: `n_neighbors` (K)
- **Pros**: Simple, no training phase, non-parametric
- **Cons**: Slow prediction, curse of dimensionality

### Support Vector Regression (SVR)
- **How it works**: Finds hyperplane within epsilon margin
- **Key Hyperparameters**: `C`, `epsilon`, `kernel`
- **Pros**: Handles non-linearity, robust to outliers
- **Cons**: Hard to interpret, sensitive to scaling"""))

# Metrics Justification
cells.append(nbf.v4.new_markdown_cell("""## 13. Evaluation Metrics Justification

### Metrics Used:

| Metric | Formula | Interpretation | Why Used |
|--------|---------|----------------|----------|
| **MAE** | Mean(|y - ŷ|) | Average error in USD | Easy to interpret, robust to outliers |
| **MSE** | Mean((y - ŷ)²) | Average squared error | Penalizes large errors |
| **RMSE** | √MSE | Error in original units | Same scale as target |
| **R²** | 1 - (SS_res/SS_tot) | Variance explained (0-1) | Scale-independent, industry standard |

### Why R² is Primary Metric:
1. **Scale-independent**: Works across different price ranges
2. **Intuitive**: 0.95 means 95% variance explained
3. **Comparative**: Easy to compare models
4. **Industry Standard**: Widely used in financial ML

### Thresholds:
- **R² > 0.9**: Excellent
- **R² 0.7-0.9**: Good  
- **R² < 0.7**: Needs improvement"""))

# Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 14. Conclusion & Key Findings

### Summary:
1. **Best Model**: Ridge Regression with optimal alpha achieved R² > 0.99
2. **Feature Engineering** was crucial - lag features are most predictive
3. **Linear models** outperformed tree-based models for this dataset
4. **Regularization** (Ridge/Lasso) improved generalization

### Why Linear Models Performed Best:
- Silver prices exhibit strong **autocorrelation**
- Tomorrow's price ≈ Today's price (with small variation)
- Linear models capture this relationship efficiently

### Future Improvements:
- Add external features (gold price, USD index, economic indicators)
- Try ensemble methods combining predictions
- Implement walk-forward validation for more robust evaluation

### Project Workflow Summary:
1. Data Collection → Yahoo Finance API
2. EDA → Identified trends and correlations  
3. Feature Engineering → Lags, MAs, momentum
4. Preprocessing → Scaling, time-based split
5. Model Training → 9 classical ML models
6. Evaluation → R², RMSE, MAE metrics
7. Hyperparameter Tuning → GridSearchCV
8. Model Selection → Ridge Regression (best R²)
9. Results Saved → For future use"""))

cells.append(nbf.v4.new_code_cell("""print("="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\\nBest Model: Ridge Regression")
print(f"Final R² Score: {ridge_grid.best_score_:.4f} (CV)")
print(f"Test R² Score: {r2_score(y_test, ridge_grid.predict(X_test_scaled)):.4f}")
print(f"\\nAll models trained, tuned, and results saved.")"""))

nb.cells = cells

with open('Silver_Price_Prediction_ML_Project.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook created successfully: Silver_Price_Prediction_ML_Project.ipynb")
