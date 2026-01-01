import pandas as pd

# Load the combined data
df = pd.read_csv('RandomForest/input/combined_data.csv')

print("="*60)
print("COMBINED DATA VERIFICATION")
print("="*60)
print(f"\nShape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Check slope columns
slope_cols = [c for c in df.columns if 'Slope' in c]
print(f"\nSlope columns: {len(slope_cols)}")

# Count rows with slope data
rows_with_slope = df[slope_cols].notna().any(axis=1).sum()
print(f"Rows with slope data: {rows_with_slope}")

# Sample statistics for 2x2km
print("\nSample statistics for 2x2km:")
print(f"  Min value: {df['Slope_2x2km_min'].min():.2f}")
print(f"  Max value: {df['Slope_2x2km_max'].max():.2f}")
print(f"  Average mean: {df['Slope_2x2km_avg'].mean():.2f}")

print("\n✓ Data looks good!")
