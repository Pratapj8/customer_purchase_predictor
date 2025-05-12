# Entire code
# 🧮 Predict purchase amount $ using gradient descent

## Features that could influence a customer's purchase amount

| TimeOnSite | PageViews | PreviousPurchases | AdClicks | IsReturningCustomer | PurchaseAmount |
| ---------- | --------- | ----------------- | -------- | ------------------- | -------------- |
| 5.1        | 12        | 0                 | 2        | 0                   | 30             |
| 7.8        | 15        | 1                 | 0        | 1                   | 75             |
| 2.5        | 5         | 0                 | 1        | 0                   | 20             |
| 9.2        | 18        | 2                 | 3        | 1                   | 120            |
| 6.5        | 14        | 1                 | 0        | 1                   | 80             |
| 3.3        | 6         | 0                 | 2        | 0                   | 35             |

- **Goal**: Predict PurchaseAmount using linear regression and gradient descent.
- The target/output we want to predict is PurchaseAmount — how much money they spent?

# Features:
- TimeOnSite (minutes): Time spent on the site.
- PageViews: Number of product pages viewed.
- PreviousPurchases: Count of previous purchases.
- AdClicks: Number of ad clicks.
- IsReturningCustomer (0 or 1): Whether the customer is returning.

| Feature               | What it Means                            |
| --------------------- | ---------------------------------------- |
| `TimeOnSite`          | How many minutes they spent browsing     |
| `PageViews`           | How many product pages they viewed       |
| `PreviousPurchases`   | Number of past purchases                 |
| `AdClicks`            | How many times they clicked on ads       |
| `IsReturningCustomer` | 1 if they bought before, 0 if first time |

# Creating a Data for e-commerce customer's and using gradient descent to train a simple linear regression model to predict how much a customer will spend (Purchase Amount).
"""

import numpy as np

# Random values
np.random.seed(42)

n_samples = 100  # 100 customers

# Generate a data for each feature
TimeOnSite = np.random.normal(6, 2, n_samples)  # Simulate time spent on site (mean=6 mins, std=2)
PageViews = np.random.randint(5, 20, n_samples)  # Random number of pages viewed (5 to 19)
PreviousPurchases = np.random.randint(0, 5, n_samples)  # Number of past purchases (0 to 4)
AdClicks = np.random.randint(0, 4, n_samples)  # Number of ad clicks (0 to 3)
IsReturningCustomer = np.random.randint(0, 2, n_samples)  # 0 = new customer, 1 = returning

# Combine all the features into one big matrix (each row = 1 customer)
X = np.column_stack((TimeOnSite, PageViews, PreviousPurchases, AdClicks, IsReturningCustomer))

X

"""# Creating output column — how much each customer spent ?"""

# We invent the "true" relationship between features and spending
true_weights = np.array([5, 3, 10, 2, 15])  # True weights for each feature
bias = 20  # Constant bias added for linear combination

# Adding a bit of random noise to make it realistic
y = X @ true_weights + bias + np.random.normal(0, 10, n_samples)

"""- Now we have inputs as X and output as y"""

y

"""# Training with Gradient Descent

"""

# Before training, we normalize the input data (makes learning faster and more stable) / mean = 0 & std =1
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

"""-  Adding a bias column to X_norm"""

# Add a bias (intercept) term to the normalized features
X_b = np.c_[np.ones((n_samples, 1)), X_norm]

"""- We initialize our model's weights / start guesses"""

# Initialize model parameters (weights) to zeros, including bias term
theta = np.zeros(X_b.shape[1])

# Gradient descent loop to update the weights
learning_rate = 0.1  # Step size for updating weights during training,A smaller value results in smaller updates, and a larger value results in larger updates
n_iterations = 1000  # Number of iterations for training the model,More iterations usually lead to better convergence


for i in range(n_iterations):
    predictions = X_b @ theta      # model prediction
    errors = predictions - y       # difference between prediction and actual
    gradients = (2/n_samples) * X_b.T @ errors  # compute gradient
    theta -= learning_rate * gradients          # update weights

# After the loop, we get the trained weights
print("Final weights:", theta)

import matplotlib.pyplot as plt

# Feature names for labeling
feature_names = ['TimeOnSite', 'PageViews', 'PreviousPurchases', 'AdClicks', 'IsReturningCustomer']
features = [TimeOnSite, PageViews, PreviousPurchases, AdClicks, IsReturningCustomer]

feature_names

features

# Ploting histograms for all features
plt.figure(figsize=(14, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.hist(features[i], bins=15, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature_names[i]}')
    plt.xlabel(feature_names[i])
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

"""- These graphs shows how many customers fall into different ranges for each feature.

# Distribution of Features (Univariate Analysis)
- 1. TimeOnSite: Roughly bell-shaped (normal-like) distribution centered around 6-7 minutes. Most users spend around 5–7 minutes of time.
- 2. PageViews: Right-skewed; many users view 5–10 web-pages, with fewer viewing more than that.
- 3. PreviousPurchases: Discrete values from 0 to 4. Most users have 0 previous purchases, but others are fairly evenly spread.
- 4. AdClicks: Discrete; 0 to 3 clicks. Most users clicked 1 or 3 ads.
- 5. IsReturningCustomer: Binary; about equal split between new and returning customers.
"""

# Ploting scatter plots: each feature vs. PurchaseAmount
plt.figure(figsize=(14, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.scatter(features[i], y, alpha=0.7)
    plt.title(f'{feature_names[i]} vs PurchaseAmount')
    plt.xlabel(feature_names[i])
    plt.ylabel('PurchaseAmount')
plt.tight_layout()
plt.show()

"""- Each dot represents one customer.

# Feature Relationships with PurchaseAmount (Bivariate Analysis)
- Checking which features influence spending the most (like more page views → higher spending?):
- 1. TimeOnSite vs PurchaseAmount:
Positive trend: More time on site generally correlates with higher purchase amount.
- 2. PageViews vs PurchaseAmount:Mild positive trend: More page views may lead to slightly higher purchases.
- 3. PreviousPurchases vs PurchaseAmount: Clear upward trend: More past purchases strongly relate to higher current purchase amount.
- 4. AdClicks vs PurchaseAmount: No clear trend: Purchase amount does not vary significantly with number of ad clicks.
- 5. IsReturningCustomer vs PurchaseAmount:Slight trend: Returning customers may tend to spend more than new ones.

---

## Insights Summary:
- Important predictors of purchase amount seem to be: PreviousPurchases, TimeOnSite, and to some extent PageViews.
- Less influential features: AdClicks and possibly IsReturningCustomer.

## Loss Tracking
"""

losses = []
for i in range(n_iterations):
    predictions = X_b @ theta
    errors = predictions - y
    loss = np.mean(errors**2)
    losses.append(loss)
    gradients = (2/n_samples) * X_b.T @ errors
    theta -= learning_rate * gradients

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Training Loss over Time")
plt.show()

"""- Mean Squared Error (MSE) loss
- The line is completely flat, meaning the loss value is not changing over iterations.
- **What this indicates :**
- 1. No learning is happening: The model's loss remains constant throughout training.
- 2. Likely reason:we are using LinearRegression from sklearn, which doesn't train iteratively — it solves for the weights in one step using a closed-form solution (not gradient descent).

# Model = LinearRegression
"""

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print("Sklearn weights:", model.coef_)
print("Sklearn bias:", model.intercept_)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y, predictions, alpha=0.7, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # perfect prediction line
plt.xlabel("Actual Purchase Amount")
plt.ylabel("Predicted Purchase Amount")
plt.title("Actual vs Predicted Purchase Amount")
plt.grid(True)
plt.tight_layout()
plt.show()

"""- **Observations:**
- Many points are close to the red line, it means model is predicting quite accurate.
"""

from sklearn.metrics import r2_score

r2 = r2_score(y, predictions)
print(f"R² Score: {r2:.4f}")

"""# Predict purchase amounts for new customer's"""

# Use the mean and std from your trained model (already computed)
# Make sure these variables are defined from your training phase
# X_mean, X_std, theta must be defined already

# Function to take user input and predict purchase amount
def predict_from_user_input():
    print("Enter the following customer data:")

    try:
        TimeOnSite = float(input("Time on site (minutes): "))
        PageViews = int(input("Number of page views: "))
        PreviousPurchases = int(input("Number of previous purchases: "))
        AdClicks = int(input("Number of ad clicks: "))
        IsReturningCustomer = int(input("Is returning customer? (0 = No, 1 = Yes): "))

        # Assemble input
        user_input = np.array([[TimeOnSite, PageViews, PreviousPurchases, AdClicks, IsReturningCustomer]])

        # Normalize using training set statistics
        user_input_norm = (user_input - X_mean) / X_std

        # Add bias term
        user_input_b = np.c_[np.ones((1, 1)), user_input_norm]

        # Predict
        predicted_purchase = user_input_b @ theta

        print(f"\n💰 Predicted Purchase Amount: ${predicted_purchase[0]:.2f}")

    except ValueError:
        print("⚠️ Invalid input. Please enter numeric values only.")

# Call the function
predict_from_user_input()

