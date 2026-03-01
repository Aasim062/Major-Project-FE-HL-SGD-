import json       # Used to read the .json data files
import pandas as pd # Used to organize data into tables (DataFrames) and save as CSV
import glob       # Used to find all files with a specific extension (like *.json) in a folder
import os         # Used to interact with the operating system (finding file paths, filenames)

# CONFIGURATION
# ---------------------------------------------------------
# Get the directory where this script is located (Data_Preprocessing folder)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# EASY INPUT: Just specify the root folder here
# The script will recursively search all subfolders for .json files
ROOT_FOLDER = os.path.join(os.path.dirname(BASE_PATH), "Dataset", "raw", "fda")

# Output folder for processed CSVs
OUTPUT_FOLDER = os.path.join(os.path.dirname(BASE_PATH), "Dataset", "processed", "fda") 

def extract_features(report):
    """
    This function looks at ONE single patient report (a dictionary) 
    and picks out only the specific info we need for our AI model.
    """
    try:
        # 1. Get the Safety Report ID (Unique ID for the report)
        # .get('key', 'default') is safer than ['key']. If the ID is missing, 
        # it won't crash the program; it will just use 'Unknown'.
        safety_id = report.get('safetyreportid', 'Unknown')
        
        # 2. Define the "Target" (What we want to predict)
        # The field 'seriousnessdeath' is '1' if the patient died.
        # We convert this to an integer: 1 means Death, 0 means Survived.
        # This creates a binary (Yes/No) target for Logistic Regression.
        death_flag = 1 if report.get('seriousnessdeath') == '1' else 0
        
        # 3. Get Patient Demographics (Age and Sex)
        # The 'patient' key contains another dictionary inside it.
        patient = report.get('patient', {})
        
        # Extract Age. If missing, it becomes None (blank).
        age = patient.get('patientonsetage', None)
        
        # Extract Sex. In FAERS, 1 usually means Male, 2 means Female.
        sex = patient.get('patientsex', None)
        
        # 4. Extract Drugs (The "Features")
        # 'drug' is a LIST of dictionaries (because a patient can take multiple drugs).
        drugs = patient.get('drug', [])
        
        # We use a "List Comprehension" here.
        # It loops through every drug in the list and grabs the 'medicinalproduct' name.
        drug_names = [d.get('medicinalproduct', '') for d in drugs]
        
        # The CSV file is a flat table, so we can't put a list in a single cell.
        # We join all drug names into one long string separated by semicolons.
        # filter(None, ...) removes any empty strings if a drug name was missing.
        # Example result: "ASPIRIN; TYLENOL; IBUPROFEN"
        drugs_str = "; ".join(filter(None, drug_names))
        
        # Return a clean dictionary with just the data we want
        return {
            'safety_report_id': safety_id,
            'age': age,
            'sex': sex,
            'drugs': drugs_str,
            'target_death': death_flag
        }
    except Exception as e:
        # If anything goes wrong with this specific report, return None.
        # This prevents one bad record from stopping the whole script.
        return None

def process_all_json_files():
    """
    This is the main function that finds files in ROOT_FOLDER and all subfolders, 
    loops through them, and saves them as CSVs.
    """
    
    # 1. Find all .json files in ROOT_FOLDER and ALL subfolders recursively
    # The ** pattern means "look in any subfolder at any depth"
    json_files = glob.glob(os.path.join(ROOT_FOLDER, "**/*.json"), recursive=True)
    
    # If the list is empty, warn the user and stop.
    if not json_files:
        print("No .json files found! Make sure you unzipped them first.")
        return

    print(f"Found {len(json_files)} JSON files to process...")

    # 2. Loop through each file found in the folder
    for filepath in json_files:
        # os.path.basename turns "C:/Users/Name/Downloads/file.json" into just "file.json"
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}...")
        
        try:
            # Open the file in 'read' mode ('r')
            # encoding='utf-8' ensures we can read special characters without crashing
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f) # Convert JSON text into a Python Dictionary
                
            # FAERS data is usually wrapped inside a "results" key.
            # We get that list. If it doesn't exist, use an empty list []
            results = data.get('results', [])
            
            # If the file is empty or has no results, skip to the next file
            if not results:
                print(f"  Warning: No 'results' found in {filename}. Skipping.")
                continue

            # 3. Extract data record by record
            extracted_data = [] # Create an empty list to hold our clean rows
            
            for report in results:
                # Run our helper function from above on this specific report
                clean_record = extract_features(report)
                
                # If the function returned valid data (not None), add it to our list
                if clean_record:
                    extracted_data.append(clean_record)
            
            # 4. Convert to DataFrame and Save
            if extracted_data:
                # pandas creates a table from our list of dictionaries
                df = pd.DataFrame(extracted_data)
                
                # Create the new filename.
                # We simply replace the .json extension with .csv
                output_filename = filename.replace('.json', '.csv')
                
                # Join the folder path and the new filename
                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                
                # Save the table to a CSV file. 
                # index=False tells pandas NOT to add a row number column (0, 1, 2...)
                df.to_csv(output_path, index=False)
                print(f"  Success! Saved {len(df)} rows to {output_filename}")
            else:
                print(f"  Warning: No valid records extracted from {filename}")
                
        except Exception as e:
            # If the file is corrupted or unreadable, print the error and keep going
            print(f"  Error processing {filename}: {e}")

# Run the main function
process_all_json_files()