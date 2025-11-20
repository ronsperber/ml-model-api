import pandas as pd
import joblib
from training.utils import get_feature_importances, labels_with_df

# Paths
MODEL_PATH = "models/iris_model.pkl"
METADATA_PATH = "metadata/iris_metadata.json"
DATA_PATH = "data/Iris.csv"

# Load the model
model = joblib.load(MODEL_PATH)

# Load the metadata
import json
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# Load the dataset (or just a few rows for testing)
df = pd.read_csv(DATA_PATH).set_index("Id")

# Prepare features (drop target)
X = df.drop(columns=["Species"])

# Predict
y_pred = model.predict(X)

# Convert predictions back to original labels
pred_labels = [metadata["classes"][i] for i in y_pred]

# For testing, print first few predictions
for i, label in zip(X.index[:10], pred_labels[:10]):
    print(f"Id {i}: Predicted Species = {label}")
label_df = labels_with_df(
    model,
    metadata,
    X
)
print(label_df.head())
# Optional: compare to true labels
true_labels = df["Species"].tolist()
accuracy = sum(p == t for p, t in zip(pred_labels, true_labels)) / len(true_labels)
print(f"Accuracy on full dataset: {accuracy:.4f}")
print("Feature importances")
feat_df = get_feature_importances(metadata)
print(feat_df)