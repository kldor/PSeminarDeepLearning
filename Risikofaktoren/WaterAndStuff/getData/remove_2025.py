import pandas as pd

csv_path = r"C:\Users\richt\Desktop\testNaturgefahrenOhneBuffer\landslides_WITH_NEW_RAIN.csv"

def main():
    print("2025 entfernen")
    
    # Laden
    df = pd.read_csv(csv_path)
    df['Datum'] = pd.to_datetime(df['Datum'])
    
    # Zählen vorher
    count_before = len(df)
    
    # Behalte nur Zeilen deren Jahr kleiner als 2025 ist
    df_clean = df[df['Datum'].dt.year < 2025]
    
    count_after = len(df_clean)
    removed = count_before - count_after
    
    print(f"Vorher: {count_before} Zeilen")
    print(f"Nachher: {count_after} Zeilen")
    print(f"Gelöscht: {removed} Zeilen (Daten aus 2025)")
    
    # Speichern
    df_clean.to_csv(csv_path, index=False)
    print("Fertig")

if __name__ == "__main__":
    main()