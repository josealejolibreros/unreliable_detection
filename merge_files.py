import os
import re
import pandas as pd

def extract_date_from_filename(filename):
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    return match.group(0) if match else "Unknown"

def process_csv_files(directory):
    all_data = []
    
    for file in os.listdir(directory):
        if "CHECKED" in file and file.endswith(".csv"):  # Filtra archivos con "CHECKED" en el nombre
            file_path = os.path.join(directory, file)
            file_date = extract_date_from_filename(file)
            
            df = pd.read_csv(file_path)
            df["file_date"] = file_date  # Agrega la fecha del archivo como nueva columna
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True, sort=False).fillna("NULLJOIN")
        combined_df.to_csv("combined_CHECKED_files.csv", index=False)
        print("Archivo combinado guardado como 'combined_CHECKED_files.csv'")
    else:
        print("No se encontraron archivos para combinar.")



if __name__ == "__main__":
    directory = "./training_data/"  # Cambia esto a tu directorio deseado
    process_csv_files(directory)

    