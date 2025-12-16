import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Step 0: Define file paths
# -----------------------------
input_file = r"C:\Users\Dell\Pictures\credit-risk-model\data\raw\data.csv"
output_dir = r"C:\Users\Dell\Pictures\credit-risk-model\data\processed"
output_file = os.path.join(output_dir, "train_with_target.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Step 1: Load dataset
# -----------------------------
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found: {input_file}")

# Ensure necessary columns exist
required_columns = ['CustomerId', 'TransactionId', 'TransactionStartTime', 'Value']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in input data: {missing_cols}")

# Ensure transaction time is datetime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# -----------------------------
# Step 2: Define snapshot date
# -----------------------------
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

# -----------------------------
# Step 3: Calculate RFM per customer
# -----------------------------
rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
    'TransactionId': 'count',                                          # Frequency
    'Value': 'sum'                                                      # Monetary
}).reset_index()

rfm.rename(columns={
    'TransactionStartTime': 'Recency',
    'TransactionId': 'Frequency',
    'Value': 'Monetary'
}, inplace=True)

# -----------------------------
# Step 4: Scale RFM features
# -----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# -----------------------------
# Step 5: Apply K-Means clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# -----------------------------
# Step 6: Identify high-risk cluster
# -----------------------------
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
high_risk_cluster = cluster_summary['Recency'].idxmax()  # highest recency = disengaged

# -----------------------------
# Step 7: Create binary target
# -----------------------------
rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)

# -----------------------------
# Step 8: Merge target into main dataset
# -----------------------------
df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# -----------------------------
# Step 9: Save updated dataset
# -----------------------------
df.to_csv(output_file, index=False)

print(f"Updated dataset saved to {output_file}")
print("High-risk distribution:")
print(rfm['is_high_risk'].value_counts())
