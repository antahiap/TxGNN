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
import pickle
import xxhash




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

def parallel_process_df(df, func, num_processes=None, tag=''):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    df_split = np.array_split(df, num_processes)
    results = []
    process_bar = f"Processing chunks {tag}"
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(func, chunk) for chunk in df_split]
        for i, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc=process_bar)):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"[!] Error in chunk {i}: {e}")
                raise

    executor.shutdown(wait=True)  
    return pd.concat(results)

def clean_selected_columns(chunk):
    return clean_columns(chunk, ['x_id', 'y_id'])


def _init_worker(node_map, node_id, lock):
        global NODE_MAP, NODE_ID, NODE_LOCK
        NODE_MAP  = node_map        # Manager().dict()
        NODE_ID   = node_id         # mp.Value("L", 0)
        NODE_LOCK = lock            # mp.Lock()

def graph_process_chunk_wrapper(chunk):
    
    global NODE_MAP, NODE_ID, NODE_LOCK
    
    edges = []
    for row in chunk.itertuples(index=False):
        x_attrs = (row.x_id, row.x_type, row.x_name, row.x_source)
        y_attrs = (row.y_id, row.y_type, row.y_name, row.y_source)

        def hash_node(attrs):
            node_key = (row.x_id, row.x_type, row.x_name, row.x_source)
            return xxhash.xxh64('|'.join(node_key)).hexdigest()

        def hash_node_slow(attrs: tuple) -> str:
            return hashlib.sha256('|'.join(map(str, attrs)).encode('utf-8')).hexdigest()

        def get_or_add_node(node_attrs, attr_dict):
            h = hash_node(node_attrs)
            with NODE_LOCK:
                if h not in NODE_MAP:
                    node_id = NODE_ID.value
                    NODE_ID.value += 1
                    NODE_MAP[h] = (node_id, attr_dict)
                else:
                    node_id = NODE_MAP[h][0]
            return node_id


        x_attr_dict = {
            'id': row.x_id, 'type': row.x_type,
            'name': row.x_name, 'source': row.x_source,
        }

        y_attr_dict = {
            'id': row.y_id, 'type': row.y_type,
            'name': row.y_name, 'source': row.y_source,
        }

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

        with mp.Pool(
            processes=num_partitions,
            initializer=_init_worker,
            initargs=(self.global_node_map, self.global_node_id, self.lock),
        ) as pool:
                edge_chunks = list(
                    tqdm(
                        pool.imap_unordered(graph_process_chunk_wrapper, df_split, chunksize=1),
                        total=len(df_split),
                        desc="Processing chunks build graph",
                    )
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
        num_processes = 6

        pkg_data = "data_primekg/03/kg.csv"
        print(f'Read {pkg_data}')
        pkg03 = pd.read_csv(self.root / Path(pkg_data), **self.opt)
        print('Make data consistent ...')
        pkg03 = parallel_process_df(pkg03, clean_selected_columns, num_processes=num_processes, tag='Ptimekg data normalization')

        synx_data = "data_synaptix/kg.csv"
        print(f'Read {synx_data}')
        synx = pd.read_csv(self.root / Path(synx_data), **self.opt).drop(columns=['x_uri', 'y_uri'], errors='ignore')
        print('Make data consistent ...')
        synx = parallel_process_df(synx, clean_selected_columns, num_processes=num_processes, tag='Synaptix data normalization')

        print('Apply merge of two kg ...')
        df = pd.concat([synx, pkg03], ignore_index=True)

        print('Make the graph')
        hg = HashGraph()
        G, _ = hg.parallel_build_graph_with_attrs(df, num_partitions=num_processes)
        G.remove_edges_from(nx.selfloop_edges(G))
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


        file_in = ''
        out_dir = self.data_path / Path(ver)
        data = kg_ne.DataPost(out_dir, file_in,  attr_list= ['id', 'type', 'name', 'source'])

        print('Write graph.pkl ...')
        graph_path = data.dir_path / Path("graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

        data.out_put_G(G)

    def read_graph_pick_to_out(self, ver):

        file_in = ''
        out_dir = self.data_path / Path(ver)
        data = kg_ne.DataPost(out_dir, file_in,  attr_list= ['id', 'type', 'name', 'source'])

        graph_path = data.dir_path / Path("graph.pkl")
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        
        file_in = ''
        out_dir = self.data_path / Path(ver + '_backup')
        data = kg_ne.DataPost(out_dir, file_in,  attr_list= ['id', 'type', 'name', 'source'])
        data.out_put_G(G)

    def ver_03_02(self):
        ver = '03_02'
        num_processes=8

        pkg_data = "data_primekg/03/kg.csv"
        print(f'Read {pkg_data}')
        pkg03 = pd.read_csv(self.root / Path(pkg_data), **self.opt)
        print('Make data consistent ...')
        pkg03 = parallel_process_df(pkg03, clean_selected_columns, num_processes=num_processes, tag='Ptimekg data normalization')

        synx_data = "data_synaptix/kg.csv"
        print(f'Read {synx_data}')
        synx = pd.read_csv(self.root / Path(synx_data), **self.opt).drop(columns=['x_uri', 'y_uri'], errors='ignore')
        print('Make data consistent ...')
        synx = parallel_process_df(synx, clean_selected_columns, num_processes=num_processes, tag='Synaptix data normalization')

        print('Apply merge of two kg ...')
        df = pd.concat([synx, pkg03], ignore_index=True)

        print('Make the graph')
        hg = HashGraph()
        G, _ = hg.parallel_build_graph_with_attrs(df, num_partitions=num_processes)
        G.remove_edges_from(nx.selfloop_edges(G))
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


        file_in = ''
        out_dir = self.data_path / Path(ver)
        data = kg_ne.DataPost(out_dir, file_in,  attr_list= ['id', 'type', 'name', 'source'])

        print('Write graph.pkl ...')
        graph_path = data.dir_path / Path("graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


        data.build_rows_parallel(G, out='kg')
        data.build_rows_parallel(G, out='edge')
        
        data.out_nodes(G)
        data.write_nodes()

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
    opt = {'nrows':100}

    synptx = Synaptix(opt=opt)
    # ver 01 and 02 are outputed from 016 notebooks
    # synptx.ver_()
    # synptx.ver_03_01() > multi id for different type
    # synptx.ver_03()
    # synptx.read_graph_pick_to_out('03')
    synptx.ver_03_02()


    # pkg = PrimeKg(opt=opt)
    # pkg.ver_02()
    # pkg.ver_03()