import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import glob
import geopandas as gpd


def load_landslide_data(shapefile_path):
    """
    Load landslide data from shapefile.
    
    Parameters:
    - shapefile_path: Path to the shapefile
    
    Returns:
    - DataFrame with landslide data
    """
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Extract coordinates from geometry
    gdf['Longitude'] = gdf.geometry.x
    gdf['Latitude'] = gdf.geometry.y
    
    # Convert to regular DataFrame
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    
    # Convert date column to datetime
    # Check for common date column names
    date_column = None
    for col in ['Datum']:
        if col in df.columns:
            date_column = col
            break
    
    if date_column is None:
        raise ValueError(f"Could not find date column in shapefile. Available columns: {df.columns.tolist()}")
    
    df['Datum'] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Select relevant columns
    base_columns = ['Datum', 'Latitude', 'Longitude','UUID', 'MOVEMENT_C', 'MOVEMENT_D']
    
    columns = base_columns.copy()
    
    result_df = df[columns].copy()
    
    print(f"Loaded {len(result_df)} landslide events")
    return result_df


def load_all_precipitation_data(data_folder):
    """
    Load all NetCDF files and create a consolidated precipitation dataset.
    Returns a dictionary with time-indexed data and coordinate information.
    """
    nc_files = sorted(glob.glob(f"{data_folder}/MERIDA_PREC_*.nc"))
    
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {data_folder}")
    
    
    all_data = []
    lat = None
    lon = None
    
    for nc_file in nc_files:
        try:
            dataset = nc.Dataset(nc_file, 'r')
            
            # Get coordinates (only once, assuming all files have same grid)
            if lat is None:
                lat = dataset.variables['lat'][:]
                lon = dataset.variables['lon'][:]
            
            # Get time and precipitation data
            time_var = dataset.variables['time']
            time_units = time_var.units if hasattr(time_var, 'units') else 'hours since 1900-01-01'
            times = nc.num2date(time_var[:], units=time_units)
            
            # Get precipitation data
            prec_var_names = ['tp']
            prec_data = None
            for var_name in prec_var_names:
                if var_name in dataset.variables:
                    prec_data = dataset.variables[var_name][:]
                    break
            
            if prec_data is None:
                # Try to find any variable with 3 dimensions (time, lat, lon)
                for var_name, var in dataset.variables.items():
                    if len(var.dimensions) == 3:
                        prec_data = var[:]
                        break
            
            if prec_data is None:
                raise ValueError(f"Could not find precipitation variable in {nc_file}")
            
            # Store data with timestamps
            # Convert cftime objects to pandas Timestamps
            for i, time_point in enumerate(times):
                # Convert cftime to standard datetime, then to pandas Timestamp
                if hasattr(time_point, 'year'):
                    # It's a cftime object, convert to datetime
                    dt = datetime(time_point.year, time_point.month, time_point.day,
                                time_point.hour, time_point.minute, time_point.second)
                    timestamp = pd.Timestamp(dt)
                else:
                    # Already a datetime object
                    timestamp = pd.Timestamp(time_point)
                
                all_data.append({
                    'time': timestamp,
                    'prec': prec_data[i, :, :]
                })
            
            dataset.close()
            
        except Exception as e:
            print(f"Error loading {nc_file}: {e}")
            continue
    
    # Sort by time
    all_data.sort(key=lambda x: x['time'])
    
    print(f"Loaded precipitation data from {all_data[0]['time']} to {all_data[-1]['time']}")

    
    return {
        'data': all_data,
        'lat': lat,
        'lon': lon
    }


def find_nearest_grid_point(target_lat, target_lon, grid_lat, grid_lon):
    """Find the nearest grid point to target coordinates."""
    lat_idx = np.argmin(np.abs(grid_lat - target_lat))
    lon_idx = np.argmin(np.abs(grid_lon - target_lon))
    return lat_idx, lon_idx


def calculate_cumulative_rainfall(landslide_date, lat, lon, prec_dataset, days_back):
    """
    Calculate cumulative rainfall for a specified period before a landslide event.
    
    Parameters:
    - landslide_date: Datetime of the landslide event
    - lat, lon: Coordinates of the landslide
    - prec_dataset: Precipitation dataset dictionary
    - days_back: Number of days to look back (7, 14, or 21)
    
    Returns:
    - Cumulative rainfall in mm, or None if no data available
    """
    lat_idx, lon_idx = find_nearest_grid_point(
        lat, lon, prec_dataset['lat'], prec_dataset['lon']
    )
    
    end_date = pd.Timestamp(landslide_date)
    start_date = end_date - timedelta(days=days_back)
    
    cumulative_prec = 0.0
    data_points = 0
    
    for entry in prec_dataset['data']:
        if start_date <= entry['time'] <= end_date:
            prec_value = entry['prec'][lat_idx, lon_idx]
            
            if not np.ma.is_masked(prec_value) and not np.isnan(prec_value):
                cumulative_prec += prec_value
                data_points += 1
    
    return cumulative_prec if data_points > 0 else None


def calculate_rainfall_for_landslides(landslides_df, prec_dataset, periods=[7, 14, 21]):
    """
    Calculate cumulative rainfall for multiple periods for all landslides.
    
    Parameters:
    - landslides_df: DataFrame with landslide data
    - prec_dataset: Precipitation dataset dictionary
    - periods: List of lookback periods in days
    
    Returns:
    - DataFrame with added rainfall columns
    """
    print("\nCalculating cumulative rainfall for each landslide event...")
    
    result_df = landslides_df.copy()
    
    for period in periods:
        rainfall_values = []
        
        for _, row in result_df.iterrows():
            date = row['Datum']
            lat = row['Latitude']
            lon = row['Longitude']
            
                        
            rainfall = calculate_cumulative_rainfall(
                date, lat, lon, prec_dataset, period
            )
            rainfall_values.append(rainfall)
            
            if rainfall is None:
                print(f"  {period}-day rainfall: No data for event on {date} at ({lat}, {lon})")
                
        
        result_df[f'Rainfall_{period}d_mm'] = rainfall_values
    
    return result_df


def print_summary_statistics(df, periods=[7, 14, 21]):
    """Print summary statistics for rainfall data."""
    print("\n=== Summary Statistics ===")
    print(f"Total landslides processed: {len(df)}")
    
    for period in periods:
        col = f'Rainfall_{period}d_mm'
        print(f"\n{period}-day cumulative rainfall:")
        print(f"  Mean:   {df[col].mean():.2f} mm")
        print(f"  Median: {df[col].median():.2f} mm")
        print(f"  Min:    {df[col].min():.2f} mm")
        print(f"  Max:    {df[col].max():.2f} mm")


def main():
    """Main function to process landslide and rainfall data."""
    # Configuration
    shapefile_path = "Risikofaktoren/DataPreperation/create_shapes_and_test_data/output/Erdrutsche213_with_random.shp"
    data_folder = "Risikofaktoren/WaterAndStuff/data"
    output_file = "Risikofaktoren/WaterAndStuff/output/landslides_with_rainfall.csv"
    
    try:
        # Load landslide data
        landslides_df = load_landslide_data(shapefile_path)
        
        # Print first few rows
        print("\nFirst few landslide records:")
        print(landslides_df.head())
        # And the last few rows
        print("\nLast few landslide records:")
        print(landslides_df.tail())
        
        # Load precipitation data
        prec_dataset = load_all_precipitation_data(data_folder)
        
        # Calculate rainfall for all periods
        result_df = calculate_rainfall_for_landslides(
            landslides_df, prec_dataset, periods=[7, 14, 21]
        )
        
        # Save results
        result_df.to_csv(output_file, index=False)
        print(f"\n\nResults saved to {output_file}")
        
        # Display statistics
        print_summary_statistics(result_df)
        
        # Display final results
        print("\n=== Final Results ===")
        print(result_df.to_string(index=False))
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
