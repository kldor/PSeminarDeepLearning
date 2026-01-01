"""
Create combined input data for RandomForest model.

This script combines data from multiple sources:
- Rainfall data (WaterAndStuff)
- Soil moisture data (SoilMoisture)
- Temperature data (Temperature)
- Ground temperature data (GroundTemperature)
- Slope statistics from raster files (Slope) for multiple polygon sizes

The output is saved to RandomForest/input/combined_data.csv
"""

import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import json
from tqdm import tqdm


def load_csv_data(base_path):
    """Load all CSV files from different data sources."""
    print("Loading CSV data from different sources...")
    
    # Define paths
    rainfall_path = base_path / "WaterAndStuff" / "output" / "landslides_with_rainfall.csv"
    soil_moisture_path = base_path / "SoilMoisture" / "output" / "landslides_with_soil_moisture.csv"
    temperature_path = base_path / "Temperature" / "output" / "landslides_with_temperature.csv"
    ground_temp_path = base_path / "GroundTemperature" / "output" / "landslides_with_ground_temperature.csv"
    
    # Load DataFrames
    df_rainfall = pd.read_csv(rainfall_path)
    df_soil = pd.read_csv(soil_moisture_path)
    df_temp = pd.read_csv(temperature_path)
    df_ground_temp = pd.read_csv(ground_temp_path)
    
    print(f"  Rainfall data: {len(df_rainfall)} rows")
    print(f"  Soil moisture data: {len(df_soil)} rows")
    print(f"  Temperature data: {len(df_temp)} rows")
    print(f"  Ground temperature data: {len(df_ground_temp)} rows")
    
    # Start with rainfall data as base
    df = df_rainfall.copy()
    
    # Merge soil moisture data
    soil_cols = ['UUID'] + [col for col in df_soil.columns if 'SoilMoisture' in col]
    df = df.merge(df_soil[soil_cols], on='UUID', how='left')
    
    # Merge temperature data
    temp_cols = ['UUID'] + [col for col in df_temp.columns if 'Temperature' in col]
    df = df.merge(df_temp[temp_cols], on='UUID', how='left')
    
    # Merge ground temperature data
    ground_temp_cols = ['UUID'] + [col for col in df_ground_temp.columns if 'GroundTemperature' in col]
    df = df.merge(df_ground_temp[ground_temp_cols], on='UUID', how='left')
    
    print(f"\nMerged data: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def calculate_slope_statistics(raster_path):
    """
    Calculate slope statistics from a raster file.
    
    Returns:
        dict: Statistics with keys 'avg', 'median', 'min', 'max'
    """
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            
            # Filter out NoData values
            nodata = src.nodata
            if nodata is not None:
                # Create mask for valid data
                valid_mask = (data != nodata) & (data > -1e30)
            else:
                valid_mask = data > -1e30
            
            valid_data = data[valid_mask]
            
            if len(valid_data) == 0:
                return {'avg': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan}
            
            return {
                'avg': float(np.mean(valid_data)),
                'median': float(np.median(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data))
            }
    except Exception as e:
        print(f"    Error processing {raster_path.name}: {e}")
        return {'avg': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan}


def extract_slope_statistics_for_all_sizes(base_path, polygon_sizes):
    """
    Extract slope statistics for all polygon sizes.
    
    Returns:
        DataFrame with UUID and slope statistics for each polygon size
    """
    print("\nExtracting slope statistics from raster files...")
    
    slope_data = []
    
    for size in polygon_sizes:
        print(f"\n  Processing {size} polygons...")
        
        # Path to valid rasters for this size
        raster_dir = base_path / "Slope" / "Output" / size / "valid_rasters"
        
        if not raster_dir.exists():
            print(f"    Directory not found: {raster_dir}")
            continue
        
        # Get all raster files
        raster_files = list(raster_dir.glob("*.tif"))
        print(f"    Found {len(raster_files)} raster files")
        
        # Process each raster
        for raster_file in tqdm(raster_files, desc=f"    {size}"):
            # Extract UUID from filename
            # Format: landslide_<UUID>.tif or no_landslide_<UUID>.tif
            filename = raster_file.stem
            if filename.startswith('landslide_'):
                uuid = filename.replace('landslide_', '')
            elif filename.startswith('no_landslide_'):
                uuid = filename.replace('no_landslide_', '')
            else:
                print(f"    Unexpected filename format: {filename}")
                continue
            
            # Calculate statistics
            stats = calculate_slope_statistics(raster_file)
            
            # Store data
            slope_data.append({
                'UUID': uuid,
                f'Slope_{size}_avg': stats['avg'],
                f'Slope_{size}_median': stats['median'],
                f'Slope_{size}_min': stats['min'],
                f'Slope_{size}_max': stats['max']
            })
    
    # Convert to DataFrame
    if not slope_data:
        print("No slope data extracted!")
        return None
    
    # Group by UUID and merge (in case there are duplicates)
    df_slope = pd.DataFrame(slope_data)
    
    # For each UUID, we should only have one entry per polygon size
    # Group by UUID and aggregate
    df_slope = df_slope.groupby('UUID').first().reset_index()
    
    print(f"\nTotal unique UUIDs with slope data: {len(df_slope)}")
    
    return df_slope


def main():
    """Main function to combine all data."""
    print("=" * 70)
    print("CREATING COMBINED INPUT DATA FOR RANDOMFOREST")
    print("=" * 70)
    
    # Define base path
    base_path = Path(__file__).parent.parent
    
    # Define polygon sizes
    polygon_sizes = [
        "5x5km", "2x2km", "1x1km", "500x500m",
        "250x250m", "100x100m", "50x50m", "30x30m", "10x10m"
    ]
    
    # Step 1: Load CSV data
    df_combined = load_csv_data(base_path)
    
    # Step 2: Extract slope statistics
    df_slope = extract_slope_statistics_for_all_sizes(base_path, polygon_sizes)
    
    if df_slope is None:
        print("\nError: No slope data found. Exiting...")
        return
    
    # Step 3: Merge slope data with combined data
    print("\nMerging slope statistics with other data...")
    df_final = df_combined.merge(df_slope, on='UUID', how='inner')
    
    print(f"  Final data: {len(df_final)} rows, {len(df_final.columns)} columns")
    
    # Step 4: Save to output
    output_dir = base_path / "RandomForest" / "input"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "combined_data.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✓ Combined data saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(df_final)}")
    print(f"Total columns: {len(df_final.columns)}")
    print(f"\nColumn groups:")
    print(f"  - Base columns: 6 (Datum, Latitude, Longitude, UUID, MOVEMENT_C, MOVEMENT_D)")
    print(f"  - Rainfall: {sum(1 for c in df_final.columns if 'Rainfall' in c)}")
    print(f"  - Soil moisture: {sum(1 for c in df_final.columns if 'SoilMoisture' in c)}")
    print(f"  - Temperature: {sum(1 for c in df_final.columns if 'Temperature' in c)}")
    print(f"  - Slope statistics: {sum(1 for c in df_final.columns if 'Slope' in c)}")
    
    # Show sample of slope statistics
    print(f"\nSample slope statistics (first row):")
    slope_cols = [c for c in df_final.columns if 'Slope' in c]
    if slope_cols:
        sample = df_final[slope_cols].iloc[0]
        for col in slope_cols[:8]:  # Show first 8 slope columns
            print(f"  {col}: {sample[col]:.2f}")
        if len(slope_cols) > 8:
            print(f"  ... and {len(slope_cols) - 8} more slope columns")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
