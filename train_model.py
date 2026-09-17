import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the canteen data
data = pd.read_csv("canteen_data.csv")

# Input features
X = data[
    [
        "attendance",
        "day",
        "exam",
        "holiday",
        "event",
        "previous_sales",
        "same_day_last_week"
    ]
]

# What we want to predict
y = data["meals_sold"]

# Convert day names into numbers
X = pd.get_dummies(X)

# Create the AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

print("AI model trained successfully!")

# Example: today's situation
test_data = pd.DataFrame({
    "attendance": [850],
    "day": ["Monday"],
    "exam": [0],
    "holiday": [0],
    "event": [0],
    "previous_sales": [800],
    "same_day_last_week": [790]
})

# Convert day
test_data = pd.get_dummies(test_data)

# Make sure test data has the same columns as training data
test_data = test_data.reindex(
    columns=X.columns,
    fill_value=0
)

# Predict meals
prediction = model.predict(test_data)

print("Predicted meals:", round(prediction[0]))