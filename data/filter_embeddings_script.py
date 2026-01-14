import numpy as np
import os

def filter_embeddings():
    # Define paths
    base_dir = "/home/jonas/Documents/Projects/sp-prediction"
    ids_file_path = os.path.join(base_dir, "data/aufgabe3/non_train_ids.txt")
    input_npz_path = os.path.join(base_dir, "data/aufgabe3/prot_t5_xl_half_uniref50-enc/reduced_30_signalP6_test_embeddings_prot_t5_xl_half_uniref50-enc.npz")
    output_npz_path = os.path.join(base_dir, "data/aufgabe3/prot_t5_xl_half_uniref50-enc/reduced_30_signalP6_test_embeddings_prot_t5_xl_half_uniref50-enc_filtered.npz")

    print(f"Reading IDs to exclude from: {ids_file_path}")
    if not os.path.exists(ids_file_path):
        print(f"Error: IDs file not found at {ids_file_path}")
        return

    with open(ids_file_path, 'r') as f:
        # distinct lines, strip whitespace
        exclude_ids = set(line.strip() for line in f if line.strip())
    
    print(f"Found {len(exclude_ids)} IDs to exclude.")

    print(f"Loading embeddings from: {input_npz_path}")
    if not os.path.exists(input_npz_path):
        print(f"Error: Input file not found at {input_npz_path}")
        return

    try:
        data = np.load(input_npz_path)
    except Exception as e:
        print(f"Error loading npz file: {e}")
        return

    # Filter entries
    filtered_data = {}
    excluded_count = 0
    kept_count = 0
    
    print("Filtering keys...")
    # Iterate over all files (keys) in the npz archive
    for key in data.files:
        # Check exact match or if key is in the set
        if key in exclude_ids:
            excluded_count += 1
        else:
            filtered_data[key] = data[key]
            kept_count += 1
            
    print(f"Filtering complete.")
    print(f"Kept: {kept_count}")
    print(f"Excluded: {excluded_count}")
    print(f"Total processed: {kept_count + excluded_count}")

    print(f"Saving filtered embeddings to: {output_npz_path}")
    try:
        np.savez_compressed(output_npz_path, **filtered_data)
        print("Successfully saved filtered file.")
    except Exception as e:
        print(f"Error saving filtered file: {e}")

if __name__ == "__main__":
    filter_embeddings()
