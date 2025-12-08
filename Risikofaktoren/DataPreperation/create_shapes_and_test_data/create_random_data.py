"""
Landslide Data Processing Script
=================================
This script performs three main tasks:
1. Generates random negative samples (non-landslide locations)
2. Creates 5x5 km square polygons around all points
3. Visualizes the results with different colors for each movement type

Input: Erdrutsche213.shp (landslide point data)
Output: 
  - Erdrutsche213_with_random.shp (points with added random data)
  - Erdrutsche213_with_random_5x5km.shp (5x5 km polygons)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# STEP 1: READ ORIGINAL DATA AND GENERATE RANDOM NEGATIVE SAMPLES
# ============================================================================

print("=" * 70)
print("STEP 1: GENERATING RANDOM NEGATIVE SAMPLES")
print("=" * 70)

# Read the original shapefile
shapefile_path = "Risikofaktoren\DataPreperation\create_shapes_and_test_data\input\Erdrutsche213.shp"
gdf_original = gpd.read_file(shapefile_path)

print(f"\nOriginal landslide data: {len(gdf_original)} points")
print(f"Columns: {gdf_original.columns.tolist()}")

# Parse dates and filter out rows before 2015-01-01
gdf_original['Datum_parsed'] = pd.to_datetime(gdf_original['Datum'], errors='coerce')
gdf_original = gdf_original[gdf_original['Datum_parsed'] >= pd.to_datetime('2015-01-01')]
print(f"After filtering for dates >= 2015-01-01: {len(gdf_original)} points")

# Get bounds of the existing data
minx, miny, maxx, maxy = gdf_original.total_bounds
print(f"\nCoordinate bounds:")
print(f"  Longitude: {minx:.6f} to {maxx:.6f}")
print(f"  Latitude: {miny:.6f} to {maxy:.6f}")

# Get date range from parsed dates
min_date = gdf_original['Datum_parsed'].min()
max_date = gdf_original['Datum_parsed'].max()
print(f"\nDate range:")
print(f"  From: {min_date}")
print(f"  To: {max_date}")

# Determine how many random samples to generate
num_random_samples = 500
print(f"\nGenerating {num_random_samples} random negative samples...")

# Generate random points
random.seed(42)  # For reproducibility
np.random.seed(42)

random_data = []

# Get max ID for creating new IDs
max_id = gdf_original['ID'].max()
max_oid = gdf_original['OID_'].max()
max_iffi = gdf_original['IFFI_Code'].max()

for i in range(num_random_samples):
    # Generate random coordinates within bounds
    rand_x = np.random.uniform(minx, maxx)
    rand_y = np.random.uniform(miny, maxy)
    
    # Generate random date within the date range
    time_delta = max_date - min_date
    random_seconds = np.random.randint(0, int(time_delta.total_seconds()) + 1)
    rand_date = min_date + timedelta(seconds=random_seconds)
    # Round to the nearest full hour
    rand_date = rand_date.replace(minute=0, second=0, microsecond=0)
    # Format to match original shapefile format: YYYY-MM-DD HH:MM:SS
    rand_date_str = rand_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create new row with random data
    new_row = {
        'ID': max_id + i + 1,
        'UUID': f'random-{i+1:04d}-no-landslide',
        'CODE': 'No Rutschung',
        'TYPE_CODE': 99.0,
        'MOVEMENT_C': 99.0,
        'MOVEMENT_D': 'No Rutschung',
        'XCoord': rand_x,
        'YCoord': rand_y,
        'OID_': max_oid + i + 1,
        'IFFI_Code': max_iffi + i + 1,
        'Datum': rand_date_str,
        'Link_First': 'N/A',
        'geometry': Point(rand_x, rand_y)
    }
    random_data.append(new_row)

# Create GeoDataFrame with random data
gdf_random = gpd.GeoDataFrame(random_data, crs=gdf_original.crs)

print(f"Generated {len(gdf_random)} random points")

# Combine original and random data
# Convert Datum_parsed back to string format to match original
gdf_original['Datum'] = gdf_original['Datum_parsed'].dt.strftime('%Y-%m-%d %H:%M:%S')
gdf_original = gdf_original.drop(columns=['Datum_parsed'])
gdf_combined = pd.concat([gdf_original, gdf_random], ignore_index=True)

print(f"\nCombined dataset: {len(gdf_combined)} points")
print(f"  - Landslides (MOVEMENT_C != 99): {len(gdf_combined[gdf_combined['MOVEMENT_C'] != 99])}")
print(f"  - No landslides (MOVEMENT_C == 99): {len(gdf_combined[gdf_combined['MOVEMENT_C'] == 99])}")

# Save combined point data
output_points = "Risikofaktoren\DataPreperation\create_shapes_and_test_data\output\Erdrutsche213_with_random.shp"
gdf_combined.to_file(output_points)
print(f"\n✓ Saved combined point shapefile to: {output_points}")

# ============================================================================
# STEP 2: CREATE 5x5 KM POLYGONS FOR ALL POINTS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: CREATING 5x5 KM POLYGONS FOR ALL POINTS")
print("=" * 70)

# Reproject to UTM for metric calculations
gdf_utm = gdf_combined.to_crs('EPSG:32632')

# Create square polygons (2500m from center = 5km total)
buffer_distance = 2500  # meters

def create_square(point, distance):
    """Create a square polygon around a point"""
    x, y = point.x, point.y
    return box(x - distance, y - distance, x + distance, y + distance)

gdf_utm['geometry'] = gdf_utm.geometry.apply(lambda pt: create_square(pt, buffer_distance))

# Convert back to original CRS
gdf_polygons = gdf_utm.to_crs('EPSG:4326')

# Verify areas
gdf_utm_temp = gdf_polygons.to_crs('EPSG:32632')
areas_km2 = gdf_utm_temp.geometry.area / 1_000_000
print(f"\nCreated {len(gdf_polygons)} polygons")
print(f"Area verification: {areas_km2.mean():.2f} km² (all should be 25 km²)")

# Save polygon shapefile
output_polygons = "Risikofaktoren\DataPreperation\create_shapes_and_test_data\output\Erdrutsche213_with_random_5x5km.shp"
gdf_polygons.to_file(output_polygons)
print(f"\n✓ Saved polygon shapefile to: {output_polygons}")

# ============================================================================
# STEP 2.5: EXPORT CSV FILES FOR BETTER OVERVIEW
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2.5: EXPORTING CSV FILES")
print("=" * 70)

# Export point data to CSV
output_points_csv = "Risikofaktoren\DataPreperation\create_shapes_and_test_data\output\Erdrutsche213_with_random_points.csv"
gdf_combined_csv = gdf_combined.copy()
# Add geometry coordinates as separate columns
gdf_combined_csv['Longitude'] = gdf_combined_csv.geometry.x
gdf_combined_csv['Latitude'] = gdf_combined_csv.geometry.y
# Drop geometry column for CSV
gdf_combined_csv_export = gdf_combined_csv.drop(columns=['geometry'])
gdf_combined_csv_export.to_csv(output_points_csv, index=False, encoding='utf-8-sig')
print(f"\n✓ Saved point data CSV to: {output_points_csv}")
print(f"  Columns: {list(gdf_combined_csv_export.columns)}")
print(f"  Rows: {len(gdf_combined_csv_export)}")

# Export polygon data to CSV with bounding box information
output_polygons_csv = "Risikofaktoren\DataPreperation\create_shapes_and_test_data\output\Erdrutsche213_with_random_polygons.csv"
gdf_polygons_csv = gdf_polygons.copy()
# Add polygon center coordinates
gdf_polygons_csv['Center_Longitude'] = gdf_polygons_csv.geometry.centroid.x
gdf_polygons_csv['Center_Latitude'] = gdf_polygons_csv.geometry.centroid.y
# Add bounding box coordinates
gdf_polygons_csv['BBox_MinX'] = gdf_polygons_csv.geometry.bounds['minx']
gdf_polygons_csv['BBox_MinY'] = gdf_polygons_csv.geometry.bounds['miny']
gdf_polygons_csv['BBox_MaxX'] = gdf_polygons_csv.geometry.bounds['maxx']
gdf_polygons_csv['BBox_MaxY'] = gdf_polygons_csv.geometry.bounds['maxy']
# Calculate area in km²
gdf_polygons_utm_temp = gdf_polygons_csv.to_crs('EPSG:32632')
gdf_polygons_csv['Area_km2'] = gdf_polygons_utm_temp.geometry.area / 1_000_000
# Drop geometry column for CSV
gdf_polygons_csv_export = gdf_polygons_csv.drop(columns=['geometry'])
gdf_polygons_csv_export.to_csv(output_polygons_csv, index=False, encoding='utf-8-sig')
print(f"\n✓ Saved polygon data CSV to: {output_polygons_csv}")
print(f"  Columns: {list(gdf_polygons_csv_export.columns)}")
print(f"  Rows: {len(gdf_polygons_csv_export)}")

# ============================================================================
# STEP 3: VISUALIZE DATA WITH DIFFERENT COLORS FOR EACH MOVEMENT TYPE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: VISUALIZING DATA")
print("=" * 70)

fig, ax = plt.subplots(figsize=(16, 11))

# Define color mapping for different movement types
landslide_mask = gdf_polygons['MOVEMENT_C'] != 99
non_landslide_mask = gdf_polygons['MOVEMENT_C'] == 99

# Get unique movement types (excluding the random ones)
movement_types = gdf_polygons[landslide_mask]['MOVEMENT_C'].unique()
print(f"\nUnique movement types found: {sorted(movement_types)}")

# Create color map for movement types
color_map = {
    1.0: '#e41a1c',  # Red
    2.0: '#377eb8',  # Blue
    3.0: '#4daf4a',  # Green
    4.0: '#984ea3',  # Purple
    5.0: '#ff7f00',  # Orange
    6.0: '#ffff33',  # Yellow
    7.0: '#a65628',  # Brown
    8.0: '#f781bf',  # Pink
    99.0: '#999999'  # Gray for no landslide
}

# Plot each movement type separately
legend_handles = []

for movement_c in sorted(gdf_polygons['MOVEMENT_C'].unique()):
    mask = gdf_polygons['MOVEMENT_C'] == movement_c
    color = color_map.get(movement_c, '#000000')
    
    if movement_c == 99.0:
        label = f'No Landslide (n={mask.sum()})'
        alpha = 0.2
        edge_alpha = 0.5
    else:
        # Get the movement description
        movement_desc = gdf_polygons[mask]['MOVEMENT_D'].iloc[0] if len(gdf_polygons[mask]) > 0 else f'Type {int(movement_c)}'
        label = f'{movement_desc} (n={mask.sum()})'
        alpha = 0.4
        edge_alpha = 0.7
    
    # Plot polygons
    gdf_polygons[mask].plot(ax=ax, color=color, edgecolor=color, 
                           alpha=alpha, linewidth=0.5, zorder=1)
    
    # Plot points on top
    gdf_combined[mask].plot(ax=ax, color=color, markersize=20, 
                           alpha=edge_alpha, zorder=5, edgecolor='black', linewidth=0.3)
    
    # Create legend handle
    legend_handles.append(mpatches.Patch(color=color, label=label, alpha=0.7))

ax.set_title('Combined Dataset: Landslides by Movement Type with 5x5 km Polygons', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.legend(handles=legend_handles, loc='best', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Total points: {len(gdf_combined)}")
print(f"  - Landslides: {len(gdf_combined[landslide_mask])}")
print(f"  - No landslides: {len(gdf_combined[non_landslide_mask])}")

# Show breakdown by movement type
print("\nBreakdown by movement type:")
for movement_c in sorted(gdf_combined['MOVEMENT_C'].unique()):
    mask = gdf_combined['MOVEMENT_C'] == movement_c
    count = mask.sum()
    if movement_c == 99.0:
        print(f"  No Landslide (99): {count}")
    else:
        movement_desc = gdf_combined[mask]['MOVEMENT_D'].iloc[0]
        print(f"  {movement_desc} ({int(movement_c)}): {count}")

print(f"\n✓ Files created:")
print(f"  1. Point shapefile: Erdrutsche213_with_random.shp")
print(f"  2. Polygon shapefile: Erdrutsche213_with_random_5x5km.shp")
print(f"  3. Point data CSV: Erdrutsche213_with_random_points.csv")
print(f"  4. Polygon data CSV: Erdrutsche213_with_random_polygons.csv")
print("=" * 70)
