# calculate_rating_diffs.py

# This script scans app CSV files to find pre/post rating blocks
# and calculates the difference in 'n' and 'p' ratings.

import os
import shutil
import pandas as pd

# --- Configurable Parameters ---
CONFIG = {
    # We use json_root_dir to find *which* participant folders to process,
    # just like in your original script.
    "json_root_dir": "/home/janus/iwso-datasets/t2-3d-body-poses",
    
    # Path to the raw CSV data
    "raw_data_dir": "/home/vault/empkins/tpD/D02/RCT/raw_data",
    
    # The final output file
    "output_file": "./rating_trend/rating_differences.csv",
    
    # A temporary directory to store merged CSVs (matches original script's behavior)
    "temp_merged_dir": "./rating_trend/temp_merged_app_csvs"
}

# Where we log data availability per participant
# STATUS_DIR = "rating_trend/rating_calc_status"

# --- Fresh start: clear output + status dirs on every run ---
def fresh_dir(path: str):
    """Removes and recreates a directory."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# def log_status_file(log_file, folder_name):
#     """Logs a folder name to a status file."""
#     os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
#     with open(log_file, 'a') as f:
#         f.write(f"{folder_name}\n")

def process_participant(folder_name, all_results):
    """
    Processes a single participant folder to find and calculate rating differences.
    """
    print(f"\nProcessing {folder_name}...")
    subject_id = folder_name.split("_")[0]

    # log_files = {
    #     "missing_app": os.path.join(STATUS_DIR, "missing_app_csv.txt"),
    #     "processed_ok": os.path.join(STATUS_DIR, "processed_ok.txt"),
    #     "no_ratings_found": os.path.join(STATUS_DIR, "no_ratings_found.txt")
    # }

    # --- Start: Logic to find/create the app_csv (from your original script) ---
    app_csv_path = os.path.join(CONFIG["raw_data_dir"], subject_id, f"{folder_name}_app.csv")
    
    # This variable will point to the CSV file we actually read
    final_csv_to_read = app_csv_path
    has_headers = False # Assume original files have no headers

    if not os.path.exists(app_csv_path):
        folder_dir = os.path.join(CONFIG["raw_data_dir"], subject_id)
        if not os.path.isdir(folder_dir):
            print(f"  ❌ Skipping {folder_name} — subject directory not found at {folder_dir}.")
            # log_status_file(log_files["missing_app"], folder_name)
            return

        # Find all 'app' CSVs in the subject's folder
        csv_files = [f for f in os.listdir(folder_dir) if "app" in f and f.endswith(".csv")]
        
        if not csv_files:
            print(f"  ❌ Skipping {folder_name} — no app CSVs found in {folder_dir}.")
            # log_status_file(log_files["missing_app"], folder_name)
            return
        
        print(f"  ℹ️ No single file. Merging {len(csv_files)} CSVs for {subject_id}...")
        try:
            dfs = []
            for f in csv_files:
                try:
                    # Read assuming no header, 3 columns, and quotes
                    temp_df = pd.read_csv(
                        os.path.join(folder_dir, f), 
                        header=None, 
                        names=['timestamp', 'label', 'value'], 
                        error_bad_lines=False,  # <-- **** FIX 1 ****
                        quoting=3 # 3 = csv.QUOTE_NONE
                    )
                    dfs.append(temp_df)
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not read {f}: {e}")

            if not dfs:
                 print(f"  ❌ Skipping {folder_name} — all app CSVs were unreadable.")
                 # log_status_file(log_files["missing_app"], folder_name)
                 return

            merged_df = pd.concat(dfs, ignore_index=True)
            
            # Clean and sort (as in original)
            merged_df['label'] = merged_df['label'].astype(str).str.strip('"').str.strip()
            merged_df['value'] = merged_df['value'].astype(str).str.strip('"').str.strip()
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'].astype(str).str.strip('"'), errors='coerce')
            merged_df = merged_df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            # Save it to the temp location (this file *will* have headers)
            final_csv_to_read = os.path.join(CONFIG["temp_merged_dir"], f"{folder_name}_merged_app.csv")
            merged_df.to_csv(final_csv_to_read, index=False, header=True)
            has_headers = True
        
        except Exception as e:
            print(f"  ❌ Failed to merge app CSVs: {e}")
            # log_status_file(log_files["missing_app"], folder_name)
            return
    # --- End: File-finding logic ---

    # --- Now, load the one `app_csv` file (either original or merged) ---
    df = None
    try:
        if has_headers:
            df = pd.read_csv(
                final_csv_to_read, 
                error_bad_lines=False, # <-- **** FIX 2 ****
                quoting=3
            )
        else:
            # Original file, no headers
            df = pd.read_csv(
                final_csv_to_read, 
                header=None, 
                names=['timestamp', 'label', 'value'], 
                error_bad_lines=False, # <-- **** FIX 2 ****
                quoting=3
            )
        
        # Clean the dataframe
        df['label'] = df['label'].astype(str).str.strip('"').str.strip()
        df['value'] = df['value'].astype(str).str.strip('"').str.strip()
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str).str.strip('"'), errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    except Exception as e:
        print(f"  ❌ Error reading final app CSV {final_csv_to_read}: {e}")
        # log_status_file(log_files["missing_app"], folder_name)
        return

    if df.empty:
         print(f"  ❌ App data is empty for {folder_name}.")
         # log_status_file(log_files["missing_app"], folder_name)
         return

    # --- Core New Logic: Find rating blocks ---
    
    # Find all 'pre-rating_start' events
    pre_rating_events = df[df['label'].str.contains("_pre-rating_start", na=False)]
    
    if pre_rating_events.empty:
        print(f"  ⚠️ No pre-rating events found for {folder_name}.")
        # log_status_file(log_files["no_ratings_found"], folder_name)
        return

    participant_results = []
    
    for idx, pre_event in pre_rating_events.iterrows():
        # 1. Identify the trial base label (e.g., 'coping_ng_02')
        base_label = pre_event['label'].replace("_pre-rating_start", "")
        
        # 2. Find the *next* 'post-rating_start' event for *this* base_label
        post_event_label = f"{base_label}_post-rating_start"
        
        # Search for post_event *after* the pre_event
        # We need to filter the DataFrame *after* the current index (idx+1)
        post_event_rows = df.loc[idx+1:] 
        post_event_row = post_event_rows[post_event_rows['label'] == post_event_label]
        
        if post_event_row.empty:
            print(f"  ⚠️ Found pre-rating for '{base_label}' but no matching post-rating.")
            continue
            
        post_event_idx = post_event_row.index[0]
        
        # 3. Define search windows
        # Pre-rating window: from *after* pre-start to *before* post-start
        pre_rating_window = df.loc[idx+1 : post_event_idx-1]
        
        # Find the '_end' event to define the end of the trial
        end_event_label = f"{base_label}_end"
        
        # Search for end event *after* the post_event
        end_event_rows = df.loc[post_event_idx+1:]
        end_event_row = end_event_rows[end_event_rows['label'] == end_event_label]
        
        post_rating_window = df.loc[post_event_idx+1:] # Default to end of file
        if not end_event_row.empty:
            end_event_idx = end_event_row.index[0]
            post_rating_window = df.loc[post_event_idx+1 : end_event_idx]

        # 4. Extract ratings
        try:
            # Find pre-ratings (n_1, p_1)
            pre_n_val = pre_rating_window[pre_rating_window['label'] == f"{base_label}_rating_n_1"]['value'].values[0]
            pre_p_val = pre_rating_window[pre_rating_window['label'] == f"{base_label}_rating_p_1"]['value'].values[0]
            
            # Find post-ratings (n_2, p_2) - based on your example
            post_n_val = post_rating_window[post_rating_window['label'] == f"{base_label}_rating_n_2"]['value'].values[0]
            post_p_val = post_rating_window[post_rating_window['label'] == f"{base_label}_rating_p_2"]['value'].values[0]

            # 5. Convert to numeric and calculate diff
            pre_n = pd.to_numeric(pre_n_val)
            pre_p = pd.to_numeric(pre_p_val)
            post_n = pd.to_numeric(post_n_val)
            post_p = pd.to_numeric(post_p_val)
            
            diff_n = post_n - pre_n
            diff_p = post_p - pre_p

            n_change_type = "neutral"
            p_change_type = "neutral"
            
            if(diff_n > 0):
                n_change_type = "positive"
            elif(diff_n < 0):
                n_change_type = "negative"

            if(diff_p > 0):
                p_change_type = "positive"
            elif(diff_p < 0):
                p_change_type = "negative"
            
            # 6. Store result
            participant_results.append({
                "subject_id": subject_id,
                "folder_name": folder_name,
                "trial_label": base_label,
                "pre_n": pre_n,
                "pre_p": pre_p,
                "post_n": post_n,
                "post_p": post_p,
                "diff_n": diff_n,
                "diff_p": diff_p, 
                "n_change_type": n_change_type,
                "p_change_type": p_change_type
            })
            
        except IndexError:
            # This means one of the ...['value'].values[0] failed (rating not found)
            print(f"  ⚠️ Missing rating (n_1, p_1, n_2, or p_2) for trial '{base_label}'.")
        except Exception as e:
            # This catches errors from pd.to_numeric (e.g., if value is not a number)
            print(f"  ❌ Error converting numbers for trial '{base_label}': {e}")
    
    if participant_results:
        all_results.extend(participant_results)
        # log_status_file(log_files["processed_ok"], folder_name)
        print(f"  ✅ Found {len(participant_results)} complete rating trials for {folder_name}.")
    else:
         print(f"  ⚠️ No *complete* rating trials found for {folder_name}.")
         # log_status_file(log_files["no_ratings_found"], folder_name)

def main():
    """
    Main function to run the data processing.
    """
    # fresh_dir(STATUS_DIR)
    fresh_dir(CONFIG["temp_merged_dir"]) # Clear temp dir
    
    # This list will hold all results (dictionaries) from all participants
    all_results = []
    
    if not os.path.isdir(CONFIG["json_root_dir"]):
        print(f"❌ Error: json_root_dir not found at {CONFIG['json_root_dir']}")
        print("Please check the 'json_root_dir' path in the CONFIG.")
        return
        
    if not os.path.isdir(CONFIG["raw_data_dir"]):
        print(f"❌ Error: raw_data_dir not found at {CONFIG['raw_data_dir']}")
        print("Please check the 'raw_data_dir' path in the CONFIG.")
        return

    # Use the same main loop as the original script
    print(f"Scanning for folders in {CONFIG['json_root_dir']}...")
    for folder in sorted(os.listdir(CONFIG["json_root_dir"])):
        if os.path.isdir(os.path.join(CONFIG["json_root_dir"], folder)):
            process_participant(folder, all_results)
    
    # --- Save all results to a single CSV ---
    if not all_results:
        print("\nNo data processed. Output file will be empty.")
        return

    print(f"\nSaving {len(all_results)} total results to {CONFIG['output_file']}...")
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for clarity
    columns = [
        "subject_id", "folder_name", "trial_label", 
        "diff_n", "diff_p", 
        "pre_n", "post_n", "pre_p", "post_p", "n_change_type", "p_change_type"
    ]
    # Ensure we only include columns that were successfully created
    final_columns = [col for col in columns if col in results_df.columns]
    results_df = results_df[final_columns]
    
    results_df.to_csv(CONFIG['output_file'], index=False)
    print("✅ Done.")
    
    # Clean up temp dir
    try:
        shutil.rmtree(CONFIG["temp_merged_dir"])
        print(f"Cleaned up temporary directory: {CONFIG['temp_merged_dir']}")
    except Exception as e:
        print(f"  ⚠️ Could not remove temp directory: {e}")

if __name__ == "__main__":
    main()
    