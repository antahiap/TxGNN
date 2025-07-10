from pathlib import Path
import pandas as pd
from synaptix_map_edges import map_e
import kg_to_node_edges as kg_ne
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing
import numpy as np
import networkx as nx
from tqdm import tqdm


def to_float(x):
    try:
        return float(x)
    except:
        return np.nan

def uppercase_string(x):
    try:
        return str(x).upper()
    except:
        return ""
    
def clean_columns(chunk, cols=['x_id', 'y_id']):
    for col in cols:
        chunk[col] = chunk[col].apply(to_float).apply(uppercase_string)
    return chunk

def parallel_process_df(df, func, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    df_split = np.array_split(df, num_processes)
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(func, chunk) for chunk in df_split]
        for i, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing chunks")):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"[!] Error in chunk {i}: {e}")
                raise

    executor.shutdown(wait=True)  
    return pd.concat(results)

def clean_selected_columns(chunk):
    return clean_columns(chunk, ['x_id', 'y_id'])

def process_chunk_wrapper(args):
    chunk, global_node_map, global_node_id, lock = args
    edges = []
    for row in chunk.itertuples(index=False):
        x_attrs = (row.x_id, row.x_type, row.x_name, row.x_source)
        y_attrs = (row.y_id, row.y_type, row.y_name, row.y_source)

        x_attr_dict = {
            'id': row.x_id, 'type': row.x_type,
            'name': row.x_name, 'source': row.x_source,
        }

        y_attr_dict = {
            'id': row.y_id, 'type': row.y_type,
            'name': row.y_name, 'source': row.y_source,
        }

        def hash_node(attrs: tuple) -> str:
            return hashlib.sha256('|'.join(map(str, attrs)).encode('utf-8')).hexdigest()

        def get_or_add_node(node_attrs, attr_dict):
            h = hash_node(node_attrs)
            with lock:
                if h not in global_node_map:
                    node_id = global_node_id.value
                    global_node_id.value += 1
                    global_node_map[h] = (node_id, attr_dict)
                else:
                    node_id = global_node_map[h][0]
            return node_id

        x_node_id = get_or_add_node(x_attrs, x_attr_dict)
        y_node_id = get_or_add_node(y_attrs, y_attr_dict)

        edge_attr = {
            'relation': row.relation,
            'display_relation': row.display_relation
        }

        edges.append((x_node_id, y_node_id, edge_attr))
    return edges



class HashGraph:

    def __init__(self):
        self.manager = mp.Manager()
        self.global_node_map = self.manager.dict()
        self.global_node_id = self.manager.Value('i', 0)
        self.lock = self.manager.Lock()

    def parallel_build_graph_with_attrs(self, df, num_partitions=None):
        print("[INFO] Starting graph construction...")

        if num_partitions is None:
            num_partitions = mp.cpu_count()

        df_split = np.array_split(df, num_partitions)
        print(f"[INFO] Split data into {num_partitions} chunks.")

        args = [(chunk, self.global_node_map, self.global_node_id, self.lock) for chunk in df_split]

        with mp.Pool(num_partitions) as pool:
            edge_chunks = list(
                tqdm(pool.imap(process_chunk_wrapper, args), total=len(df_split), desc="Processing chunks")
            )

        print("[INFO] Combining and deduplicating edges...")

        edge_set = {}
        for chunk in edge_chunks:
            for u, v, attr in chunk:
                key = (min(u, v), max(u, v), attr['relation'], attr['display_relation'])
                edge_set[key] = (u, v, attr)

        edges = list(edge_set.values())

        final_node_map = dict(self.global_node_map)
        node_items = [(node_id, attr) for _, (node_id, attr) in final_node_map.items()]

        print(f"[INFO] Creating graph with {len(node_items)} nodes and {len(edges)} edges...")

        G = nx.Graph()
        G.add_nodes_from(node_items)
        G.add_edges_from(edges)

        print("[INFO] Graph construction complete.")
        return G, final_node_map

class Synaptix:
    def __init__(self, opt, root = '../../.images/neo4j/') -> None:  

        self.root = Path(root)
        self.data_path = self.root / Path( 'data_synaptix/')
        self.opt = opt
        self.attr_list= ['id', 'type', 'name', 'source', 'uri']


    def ver_(self):
        ver = ''

        file_in = self.data_path / Path('kg_mapped_manual.csv')
        out_dir = self.data_path / Path(ver)

        print('Make data in')
        print(out_dir)
        data = kg_ne.DataPost(
            out_dir, file_in, attr_list=self.attr_list, **self.opt)

        data.make_graph_output_synaptix(opt='01')


    def ver_03_01(self):
        ver = '03_01'

        pkg03 = pd.read_csv(self.root / Path("data_primekg/03/kg.csv"), **self.opt)
        synx = pd.read_csv(self.root / Path("data_synaptix/kg.csv"), **self.opt).drop(columns=['x_uri', 'y_uri'], errors='ignore')

        combined_df = pd.concat([synx, pkg03], ignore_index=True)

        file_in = ''
        out_dir = self.data_path / Path(ver)

        data = kg_ne.DataPost(out_dir, file_in)

        attr_list= ['id', 'type', 'name', 'source']
        data.make_g(combined_df)
        data.out_put_G(data.G)

    def ver_03(self):
        ver = '03'

        pkg_data = "data_primekg/03/kg.csv"
        print(f'Read {pkg_data}')
        pkg03 = pd.read_csv(self.root / Path(pkg_data), **self.opt)
        print('Make data consistent ...')
        pkg03 = parallel_process_df(pkg03, clean_selected_columns, num_processes=4)

        synx_data = "data_synaptix/kg.csv"
        print(f'Read {synx_data}')
        synx = pd.read_csv(self.root / Path(synx_data), **self.opt).drop(columns=['x_uri', 'y_uri'], errors='ignore')
        print('Make data consistent ...')
        synx = parallel_process_df(synx, clean_selected_columns, num_processes=4)

        print('Apply merge of two kg ...')
        df = pd.concat([synx, pkg03], ignore_index=True)

        print('Make the graph')
        hg = HashGraph()
        G, _ = hg.parallel_build_graph_with_attrs(df)
        G.remove_edges_from(nx.selfloop_edges(G))
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        file_in = ''
        out_dir = self.data_path / Path(ver)

        data = kg_ne.DataPost(out_dir, file_in,  attr_list= ['id', 'type', 'name', 'source'])
        data.out_put_G(G)

class PrimeKg:
    def __init__(self, opt, root='../../.images/neo4j') -> None:  
        self.root = root
        self.data_path = Path(root) / Path( 'data_primekg/')
        self.opt = opt
        self.attr_list= ['id', 'type', 'name', 'source']
        
    def ver_02(self):
        ver = '02'

        file_in = self.data_path / Path('kg.csv')
        out_dir = self.data_path / Path(ver)

        print('Make data in')
        print(out_dir)
        data = kg_ne.DataPost(
            out_dir, file_in, attr_list=self.attr_list, **self.opt)
        data.make_g(data.kg_raw)
        data.out_put_G(data.G)

    def ver_03(self):
        # Only works for directed graph. 
        ver = '03'

        file_in = self.data_path / Path('kg.csv')
        out_dir = self.data_path / Path(ver)

        print('Make data in')
        print(out_dir)
        data = kg_ne.DataPost(
            out_dir, file_in, attr_list=self.attr_list, **self.opt)
        
        pkg = data.kg_raw

        df = pd.DataFrame.from_dict(map_e, orient='index').reset_index()
        df = df.rename(columns={'index': 'edge_type'})
        exclude_edge = df[~df['status'].str.startswith('ok')]['edge_type'].to_list()

        pkg_filtered = pkg[~pkg['relation'].isin(exclude_edge)]
        print(f'new number of edges:{len(pkg_filtered)}, from {len(pkg)}')
        
        data.kg = pkg_filtered

        attr_list= self.attr_list
        data.make_g(pkg_filtered)
        data.out_put_G(data.G)


if __name__ == '__main__':

    opt = {}
    # opt = {'nrows':100}

    synptx = Synaptix(opt=opt)
    # ver 01 and 02 are outputed from 016 notebooks
    # synptx.ver_()
    # synptx.ver_03_01() > multi id for different type
    synptx.ver_03()


    # pkg = PrimeKg(opt=opt)
    # pkg.ver_02()
    # pkg.ver_03()