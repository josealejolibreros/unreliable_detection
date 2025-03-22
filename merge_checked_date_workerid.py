import os
import re
import pandas as pd
import random
import datetime
import sys

def extract_date_from_filename(filename):
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    return match.group(0) if match else "Unknown"

def generate_random_timestamp(date_str):
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    #start_timestamp = datetime.datetime.combine(date_obj, datetime.time(0, 1)).timestamp()
    #end_timestamp = datetime.datetime.combine(date_obj, datetime.time(23, 59)).timestamp()

    start_timestamp = datetime.datetime.combine(date_obj - datetime.timedelta(days=1), datetime.time(0, 0)).timestamp()
    end_timestamp = datetime.datetime.combine(date_obj, datetime.time(15, 0)).timestamp()
    return datetime.datetime.fromtimestamp(random.uniform(start_timestamp, end_timestamp))

def process_csv_files(checked_dir, other_dir):
    all_data = []
    total_ok = 0
    
    for file in os.listdir(checked_dir):
        if "CHECKED" in file and file.endswith(".csv"):  # Filtra archivos con "CHECKED" en el nombre
            file_path = os.path.join(checked_dir, file)
            file_date = extract_date_from_filename(file)
            
            df0 = pd.read_csv(file_path)
            df1 = df0.copy()
            df1["date"] = file_date  # Agrega la fecha del archivo como nueva columna
            
            # Buscar el archivo correspondiente en el otro directorio
            df2_path = None
            for other_file in os.listdir(other_dir):
                if file_date in other_file and other_file.endswith(".csv"):
                    df2_path = os.path.join(other_dir, other_file)
                    break
            
            if df2_path:
                df2 = pd.read_csv(df2_path)
                print("file1:", file, "file:", other_file)
                df2 = df2.astype({"recording_id": str, "workunit_id": str})
                df2 = df2.astype({"mouse_move_count": int, "mutation_count": int, "focus_count": int, "recording_id": str, "workunit_id": str})
                df2['workunit_id'] = df2['workunit_id'].str.replace(r'\s+', ' ', regex=True)
                try:
                    df1 = df1.astype({"recording_id": str, "workunit_id": str})
                    df1 = df1.astype({"mouse_move_count": int, "mutation_count": int, "focus_count": int, "mutation_count": int, "recording_id": str, "workunit_id": str})
                    df1['workunit_id'] = df1['workunit_id'].str.replace(r'\s+', ' ', regex=True)
                except KeyError:
                    df1 = df1.astype({"recording_id": str, "mouse_move_count": int, "mutation_count": int, "focus_count": int})
                
                #search coincidencias en df2 for each df1 row
                for index, row in df1.iterrows():
                    is_workunit_id = True
                    try:
                        
                        match = df2[(df2["recording_id"] == row["recording_id"]) & (df2["workunit_id"] == row["workunit_id"])].copy()
                        match = df2[(df2["recording_id"] == row["recording_id"]) & (df2["workunit_id"] == row["workunit_id"]) & (df2["focus_count"] == row["focus_count"]) & (df2["mouse_move_count"] == row["mouse_move_count"]) & (df2["mutation_count"] == row["mutation_count"])].copy()
                    except KeyError: 
                        is_workunit_id = False
                        match = df2[(df2["recording_id"] == row["recording_id"]) & (df2["focus_count"] == row["focus_count"]) & (df2["mouse_move_count"] == row["mouse_move_count"]) & (df2["mutation_count"] == row["mutation_count"])].copy()

                    
                    if not match.empty:
                        df1.at[index, "merge"] = "ok"
                        df1.at[index, "start_datetime"] = match.iloc[0]["start_datetime"]
                        total_ok += 1
                        df1.at[index, "is_same_recording_id"] = match.iloc[0]["recording_id"] == row["recording_id"]
                        df1.at[index, "is_same_focus_count"] = match.iloc[0]["focus_count"] == row["focus_count"]
                        df1.at[index, "is_same_mouse_move_count"] = match.iloc[0]["mouse_move_count"] == row["mouse_move_count"]
                        df1.at[index, "is_same_mutation_count"] = match.iloc[0]["mutation_count"] == row["mutation_count"]
                        if is_workunit_id:
                            df1.at[index, "is_same_workunit_id"] = str(match.iloc[0]["workunit_id"] == row["workunit_id"])
                        else:
                            df1.at[index, "is_same_workunit_id"] = "notpresent"
                    else:
                        print(f"match not found for recording_id {row['recording_id']} en {file} vs {other_file}")#: recording_id={row['recording_id']} workunit_id={row['workunit_id']} step_id={row['step_id']} mutation_count={row['mutation_count']}")
                        df1.at[index, "merge"] = "NULLJOIN"
                        df1.at[index, "start_datetime"] = generate_random_timestamp(file_date)
            else:
                print(f"file not found in {other_dir} for date {file_date}")
                #df1["d"] = "NULLJOIN"
                #df1["e"] = "NULLJOIN"
                #df1["f"] = "NULLJOIN"
                df1.at[index, "merge"] = "NULLJOIN"
                df1["start_datetime"] = [generate_random_timestamp(file_date) for _ in range(len(df1))]
            
            all_data.append(df1)
            sys.stdout.write(f"\rProcesado {file}: {total_ok} filas con coincidencia")
            sys.stdout.flush()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        combined_df.to_csv("combined_checked_with_time_t.csv", index=False)
        print("Archivo combinado guardado como 'combined_checked_with_time.csv'")
    else:
        print("No se encontraron archivos para procesar.")

if __name__ == "__main__":
    checked_directory = "./training_data/"  #CHECKED
    other_directory = "./original_files/"  #original
    process_csv_files(checked_directory, other_directory)