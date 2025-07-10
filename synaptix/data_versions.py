from pathlib import Path
import pandas as pd
from synaptix_map_edges import map_e
import kg_to_node_edges as kg_ne
import hashlib
import multiprocessing as mp
import numpy as np
import networkx as nx
from tqdm import tqdm

# Shared state for multiprocessing
manager = mp.Manager()
global_node_map = manager.dict()
global_node_id = manager.Value('i', 0)
lock = manager.Lock()

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

        pkg03 = pd.read_csv(self.root / Path("data_primekg/03/kg.csv"), **self.opt)
        synx = pd.read_csv(self.root / Path("data_synaptix/kg.csv"), **self.opt).drop(columns=['x_uri', 'y_uri'], errors='ignore')

        df = pd.concat([synx, pkg03], ignore_index=True)
        df = df.applymap(to_float)
        df = df.astype(str).applymap(lambda x: x.lower())

        hg = HashGraph()
        G, _ = hg.parallel_build_graph_with_attrs(df)
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        file_in = ''
        out_dir = self.data_path / Path(ver)

        data = kg_ne.DataPost(out_dir, file_in)
        data.out_put_G(G)
        ## 
        # Hash mp



class HashGraph:

    def __init__(self) -> None:
        pass

    # Hash node from identifying attributes
    def hash_node(self, attrs: tuple) -> str:
        return hashlib.sha256('|'.join(map(str, attrs)).encode('utf-8')).hexdigest()

    # Register node and assign unique int ID
    def get_or_add_node(self, node_attrs: tuple, attr_dict: dict) -> int:
        h = self.hash_node(node_attrs)
        with lock:
            if h not in global_node_map:
                node_id = global_node_id.value
                global_node_id.value += 1
                global_node_map[h] = (node_id, attr_dict)
            else:
                node_id = global_node_map[h][0]
        return node_id


    def process_chunk(self, chunk):
        # Process a chunk of rows to produce edges with attributes

        edges = []
        for row in chunk.itertuples(index=False):
            # Node identity (x and y)
            x_attrs = (row.x_id, row.x_type, row.x_name, row.x_source)
            y_attrs = (row.y_id, row.y_type, row.y_name, row.y_source)

            # Full attribute dict for graph nodes
            x_attr_dict = {
                'id': row.x_id, 'type': row.x_type,
                'name': row.x_name, 'source': row.x_source,
            }

            y_attr_dict = {
                'id': row.y_id, 'type': row.y_type,
                'name': row.y_name, 'source': row.y_source,
            }

            # Get or create node integer ID
            x_node_id = self.get_or_add_node(x_attrs, x_attr_dict)
            y_node_id = self.get_or_add_node(y_attrs, y_attr_dict)

            # Edge attribute
            edge_attr = {
                'relation': row.relation,
                'display_relation': row.display_relation
            }

            edges.append((x_node_id, y_node_id, edge_attr))

        return edges

    def parallel_build_graph_with_attrs(self, df, num_partitions=None):
        # Parallel execution wrapper

        if num_partitions is None:
            num_partitions = mp.cpu_count()

        df_split = np.array_split(df, num_partitions)

        with mp.Pool(num_partitions) as pool:
            edge_chunks = list(tqdm(pool.imap(self.process_chunk, df_split), total=len(df_split), desc="Processing edges"))

        # Flatten edge list and deduplicate (based on source/target + relation if desired)
        edge_set = {}
        for chunk in edge_chunks:
            for u, v, attr in chunk:
                key = (min(u, v), max(u, v), attr['relation'], attr['display_relation'])  # avoids duplicates
                edge_set[key] = (u, v, attr)

        # Unique edges
        edges = list(edge_set.values())

        # Extract final node list
        final_node_map = dict(global_node_map)
        node_items = [(node_id, attr) for _, (node_id, attr) in final_node_map.items()]

        # Create graph
        G = nx.Graph()
        G.add_nodes_from(node_items)
        G.add_edges_from(edges)

        return G, final_node_map

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

def to_float(val):
    try:
        return str(float(val))
    except ValueError:
        return str(val)

if __name__ == '__main__':

    opt = {}
    # opt = {'nrows':10}

    synptx = Synaptix(opt=opt)
    # ver 01 and 02 are outputed from 016 notebooks
    # synptx.ver_()
    # synptx.ver_03_01() > multi id for different type
    synptx.ver_03()


    # pkg = PrimeKg(opt=opt)
    # pkg.ver_02()
    # pkg.ver_03()