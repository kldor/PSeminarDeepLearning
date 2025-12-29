import pandas as pd

csv_path = r"C:\Users\richt\Desktop\testNaturgefahrenOhneBuffer\landslides_WITH_NEW_RAIN.csv"

def main():
    print("KUMULATIV -> INTERVALLE")
    print("Rechne alle Daten (2015-2024) um...")
    
    # 1. CSV laden
    df = pd.read_csv(csv_path)
    
    # Sicherheitskopie der Spalten machen
    # .fillna(0) falls irgendwo Nichts drin steht
    rain_0_7 = df['Rainfall_7d_mm'].fillna(0).copy()
    rain_0_14 = df['Rainfall_14d_mm'].fillna(0).copy()
    rain_0_21 = df['Rainfall_21d_mm'].fillna(0).copy()
    
    # Neues Intervall 7-14 = (Summe 0-14) MINUS (Summe 0-7)
    # .clip(lower=0) verhindert negative Werte
    interval_7_14 = (rain_0_14 - rain_0_7).clip(lower=0)
    
    # Neues Intervall 14-21 = (Summe 0-21) MINUS (Summe 0-14)
    interval_14_21 = (rain_0_21 - rain_0_14).clip(lower=0)
    
    # Werte überschreiben
    df['Rainfall_14d_mm'] = interval_7_14.round(2)
    df['Rainfall_21d_mm'] = interval_14_21.round(2)
    
    # Speichern
    print(f"Speichere {len(df)} Zeilen...")
    df.to_csv(csv_path, index=False)
    
    print("\nFERTIG")
    print("Die Tabelle enthält nun:")
    print("- Rainfall_7d_mm  : Regen Tag 0 bis 7")
    print("- Rainfall_14d_mm : Regen Tag 7 bis 14")
    print("- Rainfall_21d_mm : Regen Tag 14 bis 21")

if __name__ == "__main__":
    main()