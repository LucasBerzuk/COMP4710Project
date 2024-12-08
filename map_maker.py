import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Read GeoJSON and CSV files
geojson_file = "Neighbourhood.geojson"
csv_file = "heatmapData.csv"

geo_data = gpd.read_file(geojson_file)
csv_data = pd.read_csv(csv_file)

geo_data['name'] = geo_data['name'].str.upper()
csv_data['name'] = csv_data['name'].str.upper()

# Merge GeoDataFrame with CSV DataFrame
merged_data = geo_data.merge(csv_data, on="name")

# Plot the map
fig, ax = plt.subplots(figsize=(8, 8))

# Define the green-yellow-red gradient with non-linear spacing
colors = ["green", "yellow", "red"]
nodes = [0.0, 0.05, 1.0]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", list(zip(nodes, colors)))

# Create the heatmap with the updated colormap
merged_data.plot(column="count", cmap=cmap, legend=True, ax=ax, edgecolor="black")

# Add title and remove axes
ax.set_title("Map of Crime (2013-2024)", fontsize=14)
ax.axis("off")

# Save or display the map
plt.savefig("zoning_map.png", dpi=300)
plt.show()
