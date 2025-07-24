import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Load cluster summary file
summary = pd.read_csv("hotspot_summary.csv")

# Create a folium map centered at the first cluster
start_lat = summary['Latitude'].iloc[0]
start_lon = summary['Longitude'].iloc[0]
m = folium.Map(location=[start_lat, start_lon], zoom_start=10)

# Add marker cluster
marker_cluster = MarkerCluster().add_to(m)

# Add each hotspot to map
for _, row in summary.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Cluster {int(row['Cluster'])} - Delays: {int(row['delay_count'])}",
        icon=folium.Icon(color='red')
    ).add_to(marker_cluster)

# Save map as HTML
m.save("hotspot_map.html")
print("Map saved as hotspot_map.html")
