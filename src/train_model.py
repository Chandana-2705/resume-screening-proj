import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("data/resume_dataset.csv")

# Features
X = data.drop("Score", axis=1)

# Target
y = data["Score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/candidate_model.pkl")

print("Model Trained Successfully!")