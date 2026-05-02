import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Load the BEED dataset
try:
    # Assuming BEED_Data.csv is in the current working directory
    df = pd.read_csv("BEED_Data.csv")
except FileNotFoundError:
    print("Error: BEED_Data.csv not found. Please ensure the dataset is in the correct path.")
    exit()

# --- Data Preprocessing ---
# The raw data needs to be standardized before clustering.
# Identify the feature columns (e.g., EEG readings)
# Assuming the first few columns are the features to be clustered
feature_columns = [col for col in df.columns if col.lower() not in ['id', 'label', 'subject']]
X = df[feature_columns].copy()

# Scale the features (critical for distance-based clustering methods)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data loading and scaling complete. Data dimensions:", X_scaled.shape)