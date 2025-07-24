import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import pickle

# Load vehicle data
df = pd.read_csv("vehicle_with_features.csv")

# Filter points with long time to reach stop
delay_threshold = 10  # in minutes
delays = df[df['TimeDiffToStop_min'] > delay_threshold].copy()

# Prepare location data
coords = delays[['Latitude', 'Longitude']].dropna()
coords_radians = np.radians(coords)

# Run DBSCAN using haversine distance
kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian  # 0.5 km radius

db = DBSCAN(
    eps=epsilon,
    min_samples=10,
    algorithm='ball_tree',
    metric='haversine'
).fit(coords_radians)

# Assign cluster labels
delays = delays.loc[coords.index]
delays['Cluster'] = db.labels_

# Save clustered data
delays.to_csv("delay_clusters.csv", index=False)

# Save DBSCAN model and delay threshold
with open("hotspot_dbscan_model.pkl", "wb") as f:
    pickle.dump({
        'model': db,
        'eps_km': 0.5,
        'min_samples': 10,
        'threshold_minutes': delay_threshold
    }, f)

# Generate cluster summary (centroids)
hotspots = delays[delays['Cluster'] != -1]
summary = hotspots.groupby('Cluster').agg({
    'Latitude': 'mean',
    'Longitude': 'mean'
}).reset_index()

summary['delay_count'] = hotspots.groupby('Cluster').size().values
summary.to_csv("hotspot_summary.csv", index=False)

print("Number of clusters found:", len(summary))
print("Clustered data saved to: delay_clusters.csv")
print("Cluster summary saved to: hotspot_summary.csv")
print("Model saved to: hotspot_dbscan_model.pkl")
