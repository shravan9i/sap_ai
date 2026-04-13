import os
import pandas as pd
import pickle

# =========================
# Config
# =========================
DATA_PATH = '/app/data/train.csv'
MODEL_PATH = '/app/model/model.pkl'

DT_MAX_DEPTH = int(os.getenv('DT_MAX_DEPTH', 5))

# =========================
# Load Dataset
# =========================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

if 'target' not in df.columns:
    raise ValueError("Dataset must contain 'target' column")

X = df.drop('target', axis=1)
y = df['target']

# =========================
# Train/Test Split
# =========================
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# Model Training
# =========================
from sklearn.tree import DecisionTreeRegressor

print(f"Training with max_depth={DT_MAX_DEPTH}")
model = DecisionTreeRegressor(max_depth=DT_MAX_DEPTH)

model.fit(train_x, train_y)

# =========================
# Evaluation
# =========================
score = model.score(test_x, test_y)
print(f"Test R2 Score: {score}")

# =========================
# Save Model
# =========================
os.makedirs('/app/model/', exist_ok=True)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")