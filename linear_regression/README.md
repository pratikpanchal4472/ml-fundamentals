## Learning Outcomes

1. Simple Linear Regression Fundamentals - the math and assumptions
2. Coefficient Estimation - how to find the best-fit line using "least squares method"
3. Evaluate Relationships - is the relationship strong? linear or curved?
4. Predictive Modeling - make predictions and check accuracy

## Simple Linear Regression

A method to predict a quantitative response Y based on a single predictor variable X.

There's approximately a linear relationship between X and Y (they follow a straight line pattern)

### Example

Predicting Sales from TV Advertisement Budget.

The formula:

```
Sales = β₀ + β₁(TV)
```

Where:

- `Sales` = What you're trying to predict (Y)
- `TV` = Your input/predictor (X) - TV ad budget
- `β₀` = Intercept (baseline sales with $0 TV spend)
- `β₁` = Slope (how much sales increase per $1 of TV spend)

### Formula Breakdown

The general equation is:

```
ŷᵢ = β₀ + β₁xᵢ
```

Where:

- `ŷᵢ (y-hat)` = Predicted value at observation i
- `β₀` = Intercept estimation
- `β₁` = Slope estimation
- `xᵢ` = The actual observation value

#### Translation:

If you're predicting monthly recurring revenue (MRR) from number of sales calls:

```
Predicted MRR = β₀ + β₁(Number of Sales Calls)
```

## How Do We Find β₀ and β₁?

Using `Least Squares Method`:

Minimize the sum of squared differences between:

- `yᵢ` = actual values (real sales data)
- `ŷᵢ` = predicted values (what our line predicts)

```
min Σ(yᵢ - ŷᵢ)² = min Σ(yᵢ - (β₀ + β₁xᵢ))²
```

Find the line where the total of all errors squared is as small as possible. We square the errors so positive and negative errors don't cancel out.
| | coef | std err | t-value | p-value |
|-----------|---------|----------|----------|---------|
|Intercept |β₀ 7.0326|0.458 |15.360 |0.000 |
|TV |β₁ 0.0475|0.003 |17.668 |0.000 |

---

```
R-squared: 0.612
```

What Each Column Means:

#### Coefficients (β)

β₀ = 7.0326: Baseline sales when TV budget = $0 is 7,033 units

β₁ = 0.0475: For every $1 spent on TV ads, sales increase by 0.0475 units (or $1,000 = 47.5 units)

#### Standard Error (SE)

0.458 for β₀: Uncertainty in our intercept estimate

0.003 for β₁: Uncertainty in our slope estimate

Lower is better - means more precise estimates

#### T-value

15.360 for β₀: Very high! Intercept is highly significant

17.668 for β₁: Even higher! TV strongly predicts sales

Rule: t-value > 2 or < -2 means statistically significant

#### P-value

0.000 for both: Essentially 0% chance these relationships are random

p < 0.05 is the threshold for "statistically significant"

Both predictors are highly reliable!

#### R-squared: 0.612

Means 61.2% of the variation in Sales is explained by TV budget

The other 38.8% is due to other factors (Radio ads, word-of-mouth, etc.)

## SaaS Example

| Customer Support Tickets (X) | Monthly Churn Rate % (Y) |
| ---------------------------- | ------------------------ |
| 5                            | 2.1%                     |
| 10                           | 3.8%                     |
| 15                           | 5.2%                     |
| 20                           | 6.9%                     |

```
Churn Rate = 0.5 + 0.3(Support Tickets)

β₀ = 0.5   (std err = 0.2, t = 2.5, p = 0.03)
β₁ = 0.3   (std err = 0.05, t = 6.0, p = 0.001)
R-squared = 0.85
```

### What This Tells You

- ✅ β₁ = 0.3: Each additional support ticket increases churn by 0.3%
- ✅ t-value = 6.0: Very strong relationship (way above 2)
- ✅ p = 0.001: Only 0.1% chance this is random - highly trustworthy!
- ✅ R² = 0.85: Support tickets explain 85% of churn variation

Business Action: Invest heavily in reducing support tickets - there's a clear, proven link to churn!

### Key Takeaways

- Simple Linear Regression = One input (X) predicting one output (Y) with a straight line
- Least Squares = Method to find the best-fit line by minimizing errors
- Coefficients (β₀, β₁) = The actual numbers defining your line
- T-value & P-value = Tell you if the relationship is real or just noise
- R-squared = How much of Y is explained by X

# 🐍 Python Demo - Simple Linear Regression

## 📊 The Dataset

**Advertisement Sales Dataset** with **200 observations**:

**Variables:**

1. **TV** advertising budget (in thousands of $)
2. **Radio** advertising budget (in thousands of $)
3. **Newspaper** advertising budget (in thousands of $)
4. **Sales** figures (in thousands of units) ← **This is what we're predicting**

---

## 🎯 The Demo Shows Two Approaches:

1. **Built-in functions** (easy way using libraries)
2. **Step-by-step implementation** (understanding the math)

## Approach 1️⃣: Using Built-in Functions (The Easy Way)

This is what you'll use 99% of the time in real SaaS work!

### **Using sklearn (scikit-learn)**

```python
# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the advertising data (200 rows)
df = pd.read_csv('advertising.csv')

# Define X (predictor) and y (target)
X = df[['TV']]  # Independent variable - needs to be 2D array
y = df['sales']  # Dependent variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)  # This finds β₀ and β₁ using least squares!

# Get the coefficients
print(f"Intercept (β₀): {model.intercept_}")
print(f"Slope (β₁): {model.coef_[0]}")

# Make a prediction
tv_budget = 150  # $150k TV budget
predicted_sales = model.predict([[tv_budget]])
print(f"Predicted sales for ${tv_budget}k TV: {predicted_sales[0]:.2f}k units")
```

**What this gives you:**

```
Intercept (β₀): 7.0326
Slope (β₁): 0.0475
Predicted sales for $150k TV: 14.16k units
```

**Translation:**

- With $0 TV spend, baseline sales = 7,033 units
- For every $1k spent on TV, sales increase by 47.5 units

---

### **Using statsmodels (For Detailed Statistics)**

This gives you the **full statistical output** like t-values, p-values, R-squared!

```python
import statsmodels.api as sm

# Prepare the data
X = df[['TV']]
X = sm.add_constant(X)  # Adds the intercept term
y = df['sales']

# Fit the model
model = sm.OLS(y, X).fit()  # OLS = Ordinary Least Squares

# Print the summary (this is what you saw in the PDF!)
print(model.summary())
```

**This outputs the table you saw:**

```
                coef     std err    t-value    p-value
Intercept    7.0326      0.458      15.360     0.000
TV           0.0475      0.003      17.668     0.000

R-squared: 0.612
```

---

## Approach 2️⃣: Step-by-Step Implementation (Understanding the Math)

This is how the algorithm **actually calculates** β₀ and β₁ from scratch!

### **The Formulas**

**Step 1: Calculate β₁ (slope)**

```
β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
```

**Step 2: Calculate β₀ (intercept)**

```
β₀ = ȳ - β₁x̄
```

Where:

- **x̄** = mean of X (average TV budget)
- **ȳ** = mean of y (average sales)

### **Python Implementation from Scratch**

```python
import numpy as np

# Your data
X = df['TV'].values  # TV budgets
y = df['sales'].values  # Sales

# Calculate means
x_mean = np.mean(X)  # Average TV budget
y_mean = np.mean(y)  # Average sales

# Step 1: Calculate β₁ (slope)
numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
beta_1 = numerator / denominator

# Step 2: Calculate β₀ (intercept)
beta_0 = y_mean - (beta_1 * x_mean)

print(f"β₀ (Intercept): {beta_0:.4f}")
print(f"β₁ (Slope): {beta_1:.4f}")

# Make predictions
def predict(x_value):
    return beta_0 + beta_1 * x_value

# Test it
tv_spend = 150
predicted = predict(tv_spend)
print(f"Predicted sales for ${tv_spend}k TV: {predicted:.2f}k units")
```

**Output:**

```
β₀ (Intercept): 7.0326
β₁ (Slope): 0.0475
Predicted sales for $150k TV: 14.16k units
```

---

## 📈 Visualizing the Results

```python
import matplotlib.pyplot as plt

# Scatter plot of actual data
plt.scatter(df['TV'], df['sales'], alpha=0.5, label='Actual Data')

# Plot the regression line
X_range = np.linspace(df['TV'].min(), df['TV'].max(), 100)
y_pred = beta_0 + beta_1 * X_range
plt.plot(X_range, y_pred, color='red', linewidth=2, label='Best Fit Line')

plt.xlabel('TV Advertising Budget ($1000s)')
plt.ylabel('Sales (1000 units)')
plt.title('Sales vs TV Advertising')
plt.legend()
plt.show()
```

---

## 🧮 Calculating Residuals (Errors)

**RSS** (Residual Sum of Squares):

```python
# Make predictions for all data points
y_predicted = beta_0 + beta_1 * X

# Calculate residuals (errors)
residuals = y - y_predicted

# Calculate RSS
RSS = np.sum(residuals ** 2)
print(f"RSS (Residual Sum of Squares): {RSS:.2f}")

# Individual residual for first observation
print(f"\nFirst observation:")
print(f"Actual sales: {y[0]:.2f}")
print(f"Predicted sales: {y_predicted[0]:.2f}")
print(f"Error (residual): {residuals[0]:.2f}")
```

---

## 📊 The Correlation Matrix

**correlation matrix** between all variables:

```python
# Calculate correlation matrix
correlation_matrix = df[['TV', 'radio', 'newspaper', 'sales']].corr()
print(correlation_matrix)
```

**Results:**

```
              TV     radio  newspaper    sales
TV         1.000    0.055      0.057    0.782  ← Strong!
radio      0.055    1.000      0.354    0.576
newspaper  0.057    0.354      1.000    0.228
sales      0.782    0.576      0.228    1.000
```

**What this means:**

- **TV → Sales**: 0.782 (strong positive correlation!)
- **Radio → Sales**: 0.576 (moderate correlation)
- **Newspaper → Sales**: 0.228 (weak correlation)

**SaaS Insight:** Focus marketing budget on TV ads! They have the strongest correlation with sales.

---

## 🎯 Complete SaaS Example Code

Here's a complete working example for your SaaS metrics:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example: Predict MRR from Marketing Spend
data = {
    'marketing_spend': [5, 10, 15, 20, 25, 30, 35, 40],  # in $1000s
    'mrr': [12, 18, 24, 28, 35, 38, 44, 50]  # in $1000s
}
df = pd.DataFrame(data)

# Method 1: Using sklearn
X = df[['marketing_spend']]
y = df['mrr']

model = LinearRegression()
model.fit(X, y)

print(f"Equation: MRR = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Marketing")
print(f"\nPrediction: If we spend $50k → MRR = ${model.predict([[50]])[0]:.2f}k")

# Visualize
plt.scatter(X, y, label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Marketing Spend ($1000s)')
plt.ylabel('MRR ($1000s)')
plt.legend()
plt.title('MRR vs Marketing Spend')
plt.show()
```

---

## 🔑 Key Takeaways from Python Demo

1. **sklearn.LinearRegression()** - Quick and easy for predictions
2. **statsmodels.OLS()** - Detailed statistics (t-values, p-values, R²)
3. **Manual implementation** - Understand what's happening under the hood
4. **Visualization** - Always plot your data to see the relationship!
