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
    
def clean_columns(chunk, cols=['x_id', 'y_id', 'x_name', 'y_name']):
    for col in cols:
        chunk[col] = chunk[col].apply(to_float).apply(uppercase_string)
    return chunk

def parallel_process_df(df, func, n_procs=None, tag=''):
    if n_procs is None:
        n_procs = multiprocessing.cpu_count()

    df_split = np.array_split(df, n_procs)
    results = []
    process_bar = f"Processing chunks {tag}"
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
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
    return clean_columns(chunk)

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

        def hash_node(attrs, prefix):
            if prefix == 'x':
                node_key = (row.x_id, row.x_type, row.x_name, row.x_source)
            else:
                node_key = (row.y_id, row.y_type, row.y_name, row.y_source)
            return xxhash.xxh64('|'.join(node_key).encode('utf-8')).hexdigest()

        def hash_node_slow(attrs, prefix) -> str:
            return hashlib.sha256('|'.join(map(str, attrs)).encode('utf-8')).hexdigest()

        def get_or_add_node(node_attrs, attr_dict, prefix):
            h = hash_node(node_attrs, prefix)
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

        x_node_id = get_or_add_node(x_attrs, x_attr_dict, 'x')
        y_node_id = get_or_add_node(y_attrs, y_attr_dict, 'y')

        edge_attr = {
            'relation': row.relation,
            'display_relation': row.display_relation
        }

        edges.append((x_node_id, y_node_id, edge_attr))
    return edges

def make_hash_key(side: str, attrs: tuple) -> str:
    return hashlib.sha1(f"{side}|" + "|".join(map(str, attrs)).encode('utf-8')).hexdigest()

def relabel_edges(chunk, id_map):
    return [(id_map[xk], id_map[yk], attr) for xk, yk, attr in chunk]

def collect_nodes_and_edges(chunk):

    local_nodes = {}      # hash_key → attr dict
    local_edges = []      # (x_key, y_key, edge_attr)

    for row in chunk.itertuples(index=False):
        x_attrs = (row.x_id, row.x_type, row.x_name, row.x_source)
        y_attrs = (row.y_id, row.y_type, row.y_name, row.y_source)

        x_key = make_hash_key("X", x_attrs)
        y_key = make_hash_key("Y", y_attrs)

        if x_key not in local_nodes:
            local_nodes[x_key] = dict(id=row.x_id, type=row.x_type,
                                      name=row.x_name, source=row.x_source)
        if y_key not in local_nodes:
            local_nodes[y_key] = dict(id=row.y_id, type=row.y_type,
                                      name=row.y_name, source=row.y_source)

        edge_attr = dict(relation=row.relation, display_relation=row.display_relation)
        local_edges.append((x_key, y_key, edge_attr))

    return local_nodes, local_edges

def build_graph_parallel(df: pd.DataFrame, n_procs=None):
    if n_procs is None:
        n_procs = mp.cpu_count()

    print(f"[INFO] Starting with {n_procs} workers...")

    chunks = np.array_split(df, n_procs)

    # ---------- Phase 1: collect local nodes/edges ----------
    with mp.Pool(processes=n_procs) as pool:
        results = pool.map(collect_nodes_and_edges, chunks)

    # ---------- Phase 2: merge local node maps -------------
    global_node_map = {}
    for local_nodes, _ in results:
        global_node_map.update(local_nodes)

    global_id_map = {key: idx for idx, key in enumerate(global_node_map)}

    print(f"[INFO] Total unique nodes: {len(global_id_map)}")

    # ---------- Phase 3: relabel edges ----------------------
    all_edges_raw = [edges for _, edges in results]

    with mp.Pool(processes=n_procs) as pool:
        relabeled = pool.starmap(
            relabel_edges,
            [(chunk, global_id_map) for chunk in all_edges_raw]
        )

    edges = [edge for chunk in relabeled for edge in chunk]

    G = nx.Graph()
    G.add_nodes_from(global_node_map)
    G.add_edges_from(edges)
    print("[INFO] Graph construction complete.")

    return G, global_node_map

class HashGraph:

    def __init__(self):
        pass

    def parallel_build_graph_with_attrs(self, df, n_procs=None):
        global NODE_MAP, NODE_ID, NODE_LOCK

        print("[INFO] Starting graph construction...")

        if n_procs is None:
            n_procs = mp.cpu_count() 

        df_split = np.array_split(df, n_procs)
        print(f"[INFO] Split data into {n_procs} chunks.")

        manager = mp.Manager()
        node_map = manager.dict()
        node_id = mp.Value('L', 0)
        node_lock = mp.Lock()

        with mp.Pool(
            processes=n_procs,
            initializer=_init_worker,
            initargs=(node_map, node_id, node_lock),
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

        final_node_map = dict(node_map)
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
        n_procs = 6

        pkg_data = "data_primekg/03/kg.csv"
        print(f'Read {pkg_data}')
        pkg03 = pd.read_csv(self.root / Path(pkg_data), **self.opt)
        print('Make data consistent ...')
        pkg03 = parallel_process_df(pkg03, clean_selected_columns, n_procs=n_procs, tag='Ptimekg data normalization')

        synx_data = "data_synaptix/kg.csv"
        print(f'Read {synx_data}')
        synx = pd.read_csv(self.root / Path(synx_data), **self.opt).drop(columns=['x_uri', 'y_uri'], errors='ignore')
        print('Make data consistent ...')
        synx = parallel_process_df(synx, clean_selected_columns, n_procs=n_procs, tag='Synaptix data normalization')

        print('Apply merge of two kg ...')
        df = pd.concat([synx, pkg03], ignore_index=True)

        print('Make the graph')
        hg = HashGraph()
        G, _ = hg.parallel_build_graph_with_attrs(df, n_procs=n_procs)
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
        n_procs=8

        pkg_data = "data_primekg/03/kg.csv"
        print(f'Read {pkg_data}')
        pkg03 = pd.read_csv(self.root / Path(pkg_data), **self.opt)
        print('Make data consistent ...')
        pkg03 = parallel_process_df(pkg03, clean_selected_columns, n_procs=n_procs, tag='Ptimekg data normalization')

        synx_data = "data_synaptix/kg.csv"
        print(f'Read {synx_data}')
        synx = pd.read_csv(self.root / Path(synx_data), **self.opt).drop(columns=['x_uri', 'y_uri'], errors='ignore')
        print('Make data consistent ...')
        synx = parallel_process_df(synx, clean_selected_columns, n_procs=n_procs, tag='Synaptix data normalization')

        print('Apply merge of two kg ...')
        df = pd.concat([synx, pkg03], ignore_index=True)

        print('Make the graph')
        hg = HashGraph()
        G, _ = hg.parallel_build_graph_with_attrs(df, n_procs=n_procs)
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

    def ver_03_03(self):
        ver = '03_03'
        n_procs=2

        pkg_data = "data_primekg/03/kg.csv"
        print(f'Read {pkg_data}')
        pkg03 = pd.read_csv(self.root / Path(pkg_data), **self.opt)
        print('Make data consistent ...')
        pkg03 = parallel_process_df(pkg03, clean_selected_columns, n_procs=n_procs, tag='Ptimekg data normalization')

        synx_data = "data_synaptix/kg.csv"
        print(f'Read {synx_data}')
        synx = pd.read_csv(self.root / Path(synx_data), **self.opt).drop(columns=['x_uri', 'y_uri'], errors='ignore')
        print('Make data consistent ...')
        synx = parallel_process_df(synx, clean_selected_columns, n_procs=n_procs, tag='Synaptix data normalization')

        print('Apply merge of two kg ...')
        df = pd.concat([synx, pkg03], ignore_index=True)

        print('Make the graph')
        G = build_graph_parallel(df, n_procs=n_procs)
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
    opt = {'nrows':10}

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