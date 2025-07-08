import pandas as pd
from tqdm.auto import tqdm
import networkx as nx
import re
import os
import swifter
from pathlib import Path
import ast
import sys

from source_map import sources
from constants import URI_MANUAL_MAP, EXTRACT_CONFIG, MAP_CHUNCK

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
    uri_x = row['x_id'] 
    uri_y = row['y_id']

    return pd.Series({
        'x_id': uri_x.split('/')[-1].upper(),
        'x_source': get_source(uri_x),
        'y_id': uri_y.split('/')[-1].upper(),
        'y_source': get_source(uri_y),
        'x_uri': uri_x,
        'y_uri': uri_y
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
        file_name = f"interim_result_{start}.csv"
        df = pd.read_csv(interim_dir / Path(file_name))
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
        chunk[['x_id', 'x_source', 'y_id', 'y_source', 'x_uri', 'y_uri']] = out

        file_name = f"interim_result_{i}.csv"
        chunk.to_csv(interim_dir / Path(file_name), index=False)
        kg_p.append(chunk)
    
    return kg_p

def manual_map_process(manual_map, kg):

    def get_wrong_sources(kg, tag):
        missing_source_map = []

        for i in kg[tag].unique():
            if isinstance(i, str) and i.startswith('['):
                try:
                    iList = ast.literal_eval(i)
                except (SyntaxError, ValueError):
                    print('bad data')
                    print(i)
                    break

                a = '/'.join(iList[:-1])
                if a not in missing_source_map:
                    missing_source_map.append(a)
                    print(a, iList)  # print only the latest added list
                    #break  # stop after first match


                if iList[3] == 'snomedct':  # missing in synaptix sources > Snomed
                    if not iList[-1][0].isdigit():
                        print(iList[-1])

                elif iList[3] == 'obo': # get id tags
                    if not iList[-1].startswith(('HP_', 'MONDO_', 'DOID_')):
                        print(iList[-1], 'obo')

                elif iList[3] == 'ORDO': # to Orphanet
                    if not iList[-1].startswith('Orphanet_'):
                        print(iList[-1])

                elif iList[3] == 'efo': # missing in synaptix sources
                    if not iList[-1].startswith('EFO_'):
                        print(iList[-1])

                elif iList[3] == 'node': # use merck for now
                    if not iList[-1].startswith(('PredictedLocation', 'MechanismOfActionBroadTerm', 'TissueLabel', 'CellLabel')):  # these are all types
                        print(iList[-1])
        return missing_source_map

    columns = ['x_source', 'y_source']

    for coli in columns:
        print(f'checking values in {coli}')
        missing_source_map = get_wrong_sources(kg, coli)

        if len(missing_source_map) > 0:
            for k, v in manual_map.items():
                print(f'Changing valuse of rows with {k} on {coli} column to {v}')
                kg.loc[
                    kg[coli].astype(str).str.contains(k),
                    coli
                ] = v

            get_wrong_sources(kg, coli)       
        else:
            print('Update is done.')

        print('-'*100)

        return kg



if __name__ == '__main__':
    
    print("Your message", flush=True)
    
    data_path =  Path(EXTRACT_CONFIG['data_path'])

    test_opt = "test_opt" in sys.argv[1:]   #True/False #

    if test_opt:
        data_path= data_path  / Path('test')
        print('data_path set to: ')
        print(data_path )

    sources = [line.upper() for line in sources.split('\n')[0::3]]
    kg_file = Path(EXTRACT_CONFIG['out_name'])
    kg = pd.read_csv(data_path / kg_file, delimiter=',')#, nrows=10)


    tqdm.pandas()
    interim_dir = data_path / Path('backup')
    interim_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = MAP_CHUNCK

    kg_p = map_source_name(kg, chunk_size, interim_dir)
    final_result = pd.concat(kg_p).reset_index(drop=True)
    final_result.to_csv(data_path / Path('kg_mapped.csv'), sep=",", index=False, quoting=1)

    #---------------------
    #  as all the extraction doesn't work for all sources, in ./notebooks/16_synaptix_post.ipynb
    #  the mapped data is processed and the following additional mapping is set.


    final_result = manual_map_process(URI_MANUAL_MAP, final_result)
    final_result.to_csv(data_path / Path('kg_mapped_manual.csv'), sep=",", index=False, quoting=1)

