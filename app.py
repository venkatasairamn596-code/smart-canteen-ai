from flask import Flask, request, jsonify, render_template
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# -----------------------------
# Load training data
# -----------------------------

data = pd.read_csv("canteen_data.csv")

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

y = data["meals_sold"]

# Convert day into numerical columns
X = pd.get_dummies(X)

# -----------------------------
# Train AI model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("AI model trained successfully!")


# -----------------------------
# Create SQLite database
# -----------------------------

def create_database():

    conn = sqlite3.connect("canteen.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            attendance INTEGER,
            predicted_meals INTEGER,
            recommended_meals INTEGER,
            actual_sales INTEGER,
            food_waste INTEGER
        )
    """)

    conn.commit()
    conn.close()


create_database()


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------
# AI Prediction
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data_input = request.json

    test_data = pd.DataFrame({
        "attendance": [int(data_input["attendance"])],
        "day": [data_input["day"]],
        "exam": [int(data_input["exam"])],
        "holiday": [int(data_input["holiday"])],
        "event": [int(data_input["event"])],
        "previous_sales": [int(data_input["previous_sales"])],
        "same_day_last_week": [int(data_input["same_day_last_week"])]
    })

    # Convert day into same format as training data
    test_data = pd.get_dummies(test_data)

    # Make columns exactly match training data
    test_data = test_data.reindex(
        columns=X.columns,
        fill_value=0
    )

    # AI prediction
    prediction = model.predict(test_data)

    predicted_meals = round(prediction[0])

    # Add 5% safety buffer
    recommended_meals = round(
        predicted_meals * 1.05
    )

    return jsonify({
        "predicted_meals": predicted_meals,
        "recommended_meals": recommended_meals
    })


# -----------------------------
# Save daily record
# -----------------------------

@app.route("/save", methods=["POST"])
def save_record():

    data_input = request.json

    conn = sqlite3.connect("canteen.db")

    conn.execute("""
        INSERT INTO daily_records
        (
            date,
            attendance,
            predicted_meals,
            recommended_meals,
            actual_sales,
            food_waste
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data_input["date"],
        int(data_input["attendance"]),
        int(data_input["predicted_meals"]),
        int(data_input["recommended_meals"]),
        int(data_input["actual_sales"]),
        int(data_input["food_waste"])
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "message": "Daily record saved successfully!"
    })


# -----------------------------
# History page
# -----------------------------

@app.route("/history")
def history_page():

    return render_template("history.html")


# -----------------------------
# History data API
# -----------------------------

@app.route("/history-data")
def history_data():

    conn = sqlite3.connect("canteen.db")

    records = conn.execute("""
        SELECT
            date,
            attendance,
            predicted_meals,
            recommended_meals,
            actual_sales,
            food_waste
        FROM daily_records
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    result = []

    for record in records:

        result.append({
            "date": record[0],
            "attendance": record[1],
            "predicted_meals": record[2],
            "recommended_meals": record[3],
            "actual_sales": record[4],
            "food_waste": record[5]
        })

    return jsonify(result)


# -----------------------------
# Dashboard page
# -----------------------------

@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")


# -----------------------------
# Run Flask application
# -----------------------------

if __name__ == "__main__":

    app.run(debug=True)