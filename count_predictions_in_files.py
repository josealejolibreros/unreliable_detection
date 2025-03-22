import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np


def extract_date_from_filename(filename):
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    return match.group(0) if match else "Unknown"

def sort_arrs(arr_amount_unreliables, arr_dates):
    # Convert lists to NumPy arrays
    arr_dates = np.array(arr_dates)
    arr_amounts = np.array(arr_amount_unreliables)

    # Convert date strings to datetime objects for sorting
    date_objects = np.array(arr_dates, dtype="datetime64")

    # Get sorted indices
    sorted_indices = np.argsort(date_objects)

    # Sort both arrays
    sorted_dates = arr_dates[sorted_indices]  # Now works correctly
    sorted_amounts = arr_amounts[sorted_indices]  # Now works correctly

    return sorted_amounts, sorted_dates

def plot_dates_amount(arr_amount_unreliables, arr_dates):
    fig, ax = plt.subplots()

    ax.bar(arr_dates, arr_amount_unreliables)
    ax.set_ylabel('Unreliables', fontname="Arial")
    ax.set_title('Unreliables each iteration', fontname="Arial")
    plt.xticks(rotation=75)
    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.tick_params(axis='x', which='minor', labelsize=5)
    plt.savefig('unrealibles_each_iteration.png', dpi=300)
    plt.show()
    

def process_csv_files(directory):
    matching_count = 0
    total_unrel_predictions = 0
    total_files = 0
    arr_amount_unreliables = []
    arr_dates = []
    
    
    for file in os.listdir(directory):
        if "CHECKED" in file and file.endswith(".csv"):  # Filtra archivos con "CHECKED" en el nombre
            file_path = os.path.join(directory, file)
            file_date = extract_date_from_filename(file)
            df = pd.read_csv(file_path)
            total_files +=1
            
            if "prediction" in df.columns and "class" in df.columns:
                # Cuenta los casos donde prediction es "car" y class es "unrel"
                matching_count += ((df["prediction"] == "Reliable") & (df["class"] == "Unreliable")).sum()
            # Cuenta todos los casos donde class es "unrel"
            unreliables = (df["class"] == "Unreliable").sum()
            arr_amount_unreliables.append(unreliables)
            arr_dates.append(file_date)
            total_unrel_predictions += unreliables

    return matching_count, total_unrel_predictions, total_files, arr_amount_unreliables, arr_dates

if __name__ == "__main__":
    directory = "./training_data/"  # Cambia esto a tu directorio deseado
    matching_count, total_unrel_predictions, total_files, arr_amount_unreliables, arr_dates = process_csv_files(directory)
    
    arr_amount_unreliables, arr_dates = sort_arrs(arr_amount_unreliables, arr_dates)

    
    
    print(f"Total files: {total_files}")
    print(f"Cases where prediction es 'rel' y class es 'unrel': {matching_count}")
    print(f"Total de registros donde class es 'unrel': {total_unrel_predictions}")

    plot_dates_amount(arr_amount_unreliables, arr_dates)

    