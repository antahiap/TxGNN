import pandas as pd
from tqdm.auto import tqdm
import networkx as nx
import re
import os
import swifter

from source_map import sources

def get_source(uri):
    uri_split = uri.split('/')
    for i in range(-2, -4, -1):
        check = uri_split[i].upper()
        pattern = f'.*{re.escape(check)}.*'
        for src in sources:
            if re.match(pattern, src, re.IGNORECASE):
                return src
    return(uri_split)

def process_uri(row):
    uri_x = row['x_index'] 
    uri_y = row['y_index']

    return pd.Series({
        'x_index': uri_x.split('/')[-1].upper(),
        'x_source': get_source(uri_x),
        'y_index': uri_y.split('/')[-1].upper(),
        'y_source': get_source(uri_y),
    })

def get_existing_chunk_ranges(interim_dir="."):
    """
    Return list of (start, end) ranges of existing chunks.
    """
    ranges = []
    for fname in os.listdir(interim_dir):
        if fname.startswith("interim_result_") and fname.endswith(".csv"):
            try:
                i = int(fname.replace("interim_result_", "").replace(".csv", ""))
                # Read file just enough to get row count
                n_rows = sum(1 for _ in open(os.path.join(interim_dir, fname))) - 1  # minus header
                ranges.append((i, i + n_rows))
            except Exception:
                continue
    return sorted(ranges)

def is_range_covered(i_start, i_end, existing_ranges):
    """
    Check if [i_start, i_end) is already covered by any existing range.
    """
    for start, end in existing_ranges:
        if i_start >= start and i_end <= end:
            return True
    return False

def merge_existing_chunks(interim_dir="."):
    """
    Process remaining chunks of kg, skip already covered ranges.
    """
    tqdm.pandas()
    
    existing_ranges = get_existing_chunk_ranges(interim_dir)
    results = []
    
    # Merge existing first
    merged_existing = []
    for start, _ in existing_ranges:
        df = pd.read_csv(os.path.join(interim_dir, f"interim_result_{start}.csv"))
        merged_existing.append(df)
    
    if merged_existing:
        results.append(pd.concat(merged_existing).reset_index(drop=True))

    return results, existing_ranges

def map_source_name(kg, chunk_size, interim_dir):
    kg_p, existing_ranges = merge_existing_chunks(interim_dir)

    # Now process remaining
    for i in range(0, len(kg), chunk_size):
        i_end = min(i + chunk_size, len(kg))
        if is_range_covered(i, i_end, existing_ranges):
            print(f"✅ Skipping chunk {i}-{i_end} (already processed)")
            continue
        
        print(f"⚡ Processing chunk {i}-{i_end}...")
        chunk = kg.iloc[i:i_end].copy()
        out = chunk.progress_apply(process_uri, axis=1, result_type='expand')
        chunk[['x_index', 'x_source', 'y_index', 'y_source']] = out
        chunk.to_csv(os.path.join(interim_dir, f"interim_result_{i}.csv"), index=False)
        kg_p.append(chunk)



if __name__ == '__main__':

    
    print("Your message", flush=True)

    sources = [line.upper() for line in sources.split('\n')[0::3]]
    data_path_synaptix = '/home/apakiman/Repo/merck_gds_explr/.images/neo4j/data_synaptix/'

    kg = pd.read_csv(data_path_synaptix +'kg_keep.csv', delimiter='\t')#, nrows=10)


    tqdm.pandas()
    interim_dir = data_path_synaptix + '/backup'
    chunk_size = int(1e6)  
    # 1e4  4s > 93 min

    kg_p = map_source_name(kg, chunk_size, interim_dir)
    final_result = pd.concat(kg_p).reset_index(drop=True)
    final_result.to_csv(data_path_synaptix + 'kg_test_merg.csv', sep="\t", index=False, quoting=1)

