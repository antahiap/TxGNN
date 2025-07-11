import hashlib
import os
import sys
import networkx as nx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools
from multiprocess import Pool, cpu_count 
from functools import partial

try:
    from constants import EXTRACT_CONFIG, DATA_VER, KG_RAW
except:
    from synaptix.constants import EXTRACT_CONFIG, DATA_VER, KG_RAW

class DataPost:
    def __init__(self, dir_path, file_in, 
                 delimiter=',',
                 node_file_name = 'node.csv',
                 edge_file_name = 'edges.csv',
                 kg_file_name = 'kg.csv',
                 nrows=None,
                 skiprows=None,
                 attr_list = ['id', 'type', 'name', 'source', 'uri']
                 ): 
        
        self.dir_path = Path (dir_path)
        if not os.path.isdir(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        self.file_in = file_in 
        if file_in:
            print('Read kg input ...')
            if nrows:
                if skiprows:
                    self.kg_raw = pd.read_csv(self.file_in, delimiter=delimiter, nrows=nrows, skiprows=skiprows)
                self.kg_raw = pd.read_csv(self.file_in, delimiter=delimiter, nrows=nrows)
            else:
                self.kg_raw = pd.read_csv(self.file_in, delimiter=delimiter)
            print(file_in)
        else:
            self.kg = None

        self.nodes = None
        self.edges = None
        self.G = None
        self.kg = None

        self.node_file_name = node_file_name
        self.edge_file_name = edge_file_name
        self.kg_file_name = kg_file_name
        self.attr_list = attr_list

    def read_nodes(self, delimiter=','):
        print('Read nodes ... ')
        in_file = self.dir_path / Path(self.node_file_name)
        self.nodes = pd.read_csv (in_file, delimiter=delimiter)

    def read_edges(self, delimiter=','):
        print('Read edges ... ')
        in_file = self.dir_path / Path(self.edge_file_name)
        self.edges = pd.read_csv (in_file, delimiter=delimiter)

    def read_kg(self, delimiter=','):
        print('Read kg.csv ... ')
        in_file = self.dir_path / Path(self.kg_file_name)

        self.kg = pd.read_csv(in_file, delimiter=delimiter)

    def write_nodes(self,delimiter=','):
        print('Write node.csv ... ')
        out_file = self.dir_path / Path(self.node_file_name)
        self.nodes.to_csv(out_file, sep=delimiter, index=False, quoting=1)
        print(out_file)

    def write_edges(self, delimiter=','):
        print('Write edges.csv ... ')
        out_file = self.dir_path / Path(self.edge_file_name)
        self.edges.to_csv(out_file, sep=delimiter, index=False)
        print(out_file)

    def write_kg(self, delimiter=','):
        print('Write kg.csv ...')
        
        out_file = self.dir_path / Path(self.kg_file_name)
        self.kg.to_csv(out_file, sep=delimiter, index=False)
        print(out_file)

    def make_g(self, kg):
        print('Load KG to a graph')
        
        def add_nodes_from_df(df, node_prefixes, col_list):
            print('... Add nodes')
            G = nx.MultiDiGraph()
            seen_nodes = {}

            for prefix in node_prefixes:
                index_col = f"{prefix}_index"
                type_col = f"{prefix}_type"
                cols = [f'{prefix}_{col}' for col in col_list]
                attr_names = dict(zip(cols, col_list))

                # Drop duplicate node entries based on node_index
                node_df = df.drop_duplicates(subset=index_col)

                # Pre-collect node data: (node_index, attr_dict)
                nodes_to_add = []
                print(f'⏳ Processing nodes for prefix "{prefix}" ({len(node_df)} entries)...')
                for row in tqdm(node_df.itertuples(index=False), total=len(node_df), desc=f'Adding {prefix} nodes'):
           
                # for row in node_df.itertuples(index=False):
                    row_dict = row._asdict()
                    node_index = row_dict[index_col]
                    attributes = {attr_names[col]: row_dict.get(col) for col in cols if col in row_dict}

                    node_type = row_dict.get(type_col)
                    if node_type:
                        attributes["type"] = node_type

                    if node_index in seen_nodes:
                        existing_type = seen_nodes[node_index]
                        if existing_type != node_type:
                            raise ValueError(
                                f"[Error] Node ID '{node_index}' already exists with type '{existing_type}', "
                                f"but new type is '{node_type}'. Type mismatch detected."
                            )
                        continue

                    seen_nodes[node_index] = node_type
                    nodes_to_add.append((node_index, attributes))

                # Add all nodes in batch
                G.add_nodes_from(nodes_to_add)

            return G 

        self.G = add_nodes_from_df(kg, node_prefixes=['x', 'y'], col_list=self.attr_list)

        # Add edges
        print('... Add edges')


        def add_edges_with_attrs(G, kg):

            def make_attrs(relation, display_relation):
                return {'relation': relation, 'display_relation': display_relation}
            
            kg_unique = kg.drop_duplicates(subset=['x_index', 'y_index', 'relation', 'display_relation'])

            edge_tuples = zip(
                kg_unique['x_index'].values,
                kg_unique['y_index'].values,
                itertools.starmap(make_attrs, zip(kg_unique['relation'].values, kg_unique['display_relation'].values))
            )
            print(f"... Adding {len(kg_unique):,} edges")
            G.add_edges_from(edge_tuples)

            return G
        
        self.G = add_edges_with_attrs(self.G, kg)

    def get_largest_g(self):

        components = list(nx.connected_components(self.G))
        largest_cc = max(components, key=len)
        original_nodes = self.G.number_of_nodes()
        original_edges = self.G.number_of_edges()

        # Create a subgraph with only the nodes in the largest component
        G_largest = self.G.subgraph(largest_cc).copy()

        # Count remaining nodes and edges
        remaining_nodes = G_largest.number_of_nodes()
        remaining_edges = G_largest.number_of_edges()

        # Calculate deletion ratios
        del_nodes = (original_nodes - remaining_nodes) 
        del_edges = original_edges - remaining_edges
        deleted_nodes_pct = (del_nodes/ original_nodes) * 100
        deleted_edges_pct = (del_edges / original_edges) * 100

        # Print results
        print(f"To be Deleted Nodes: {deleted_nodes_pct:.2f}%, {del_nodes}")
        print(f"To be Deleted Edges: {deleted_edges_pct:.2f}%", {del_edges})
        
        return G_largest
    
    def sort_nodes(self, G_in, tag_sort='type', neo4j_map_out='neo4j_ids_mapping.json'):

        sorted_nodes = sorted(G_in.nodes(), key=lambda n: G_in.nodes[n][tag_sort])
        mapping = {node: idx for idx, node in enumerate(sorted_nodes)}
        G_renumbered_0 = nx.relabel_nodes(G_in, mapping)
        G_renumbered =  nx.convert_node_labels_to_integers(G_renumbered_0, first_label=0, ordering='default')
    
        # neo4j_ids_mapping = dict(zip(G_renumbered.nodes(), G_renumbered_0.nodes()))

        # with open(self.dir_path + neo4j_map_out, 'w') as f:
        #     json.dump(neo4j_ids_mapping, f, indent=2)
        return G_renumbered_0

    def out_put_G(self, G):

        self.out_kg(G)
        self.write_kg()

        self.out_nodes(G)
        self.write_nodes()

        self.out_edges(G)
        self.write_edges()

    def out_nodes(self, G,
                  prefix='node'):
        print('Process nodes ... ')
        node_rows = []
        for node_index, attrs in G.nodes(data=True):
            row = {f'{prefix}_index': node_index}
            for attr in self.attr_list:
                row[f'{prefix}_{attr}'] = attrs.get(attr, '')
            
            node_rows.append(row)

        self.nodes = pd.DataFrame(node_rows)

    def out_edges(self, G):
        # Efficient extraction using list comprehension for edges

        print('Process edges ... ')
        edge_rows = []
        for u, v, data in G.edges(data=True):
            row = {
                'relation': data.get('relation', ''),
                'display_relation': data.get('display_relation', ''),
                'x_index':u,  # Use node ID for x_index
                'y_index':v  # Use node ID for y_index
            }
            edge_rows.append(row)
        self.edges = pd.DataFrame(edge_rows)
        
    def out_kg(self, G):

        print('Process kg ... ')
        # Collect rows for CSV
        rows = []

        for u, v, data in G.edges(data=True):
            x = G.nodes[u]
            y = G.nodes[v]

            row = {
                'relation': data.get('relation', ''),
                'display_relation': data.get('display_relation', '')
            }
            row['x_index'] = u
            for attr in self.attr_list:
                row[f'x_{attr}'] = x.get(attr, '')
            
            row['y_index'] = v
            for attr in self.attr_list:
                row[f'y_{attr}'] = y.get(attr, '')

            rows.append(row)

        self.kg = pd.DataFrame(rows)

    def _edge_to_edgerow(self, edge, node_attrs):

        u, v, data = edge

        row = {
            'relation': data.get('relation', ''),
            'display_relation': data.get('display_relation', ''),
            'x_index':u,  # Use node ID for x_index
            'y_index':v  # Use node ID for y_index
        }

        return row 
    
    def _edge_to_kgrow(self, edge, node_attrs):

        u, v, data = edge
        x, y = node_attrs[u], node_attrs[v]

        row = {
            'relation': data.get('relation', ''),
            'display_relation': data.get('display_relation', '')
        }
        row['x_index'] = u
        for attr in self.attr_list:
            row[f'x_{attr}'] = x.get(attr, '')
            
        row['y_index'] = v
        for attr in self.attr_list:
            row[f'y_{attr}'] = y.get(attr, '')

        return row 
    
    def build_rows_parallel(
        self,
        G,
        out='kg',
        chunk_size: int = 1_000_000,
        pool_chunksize: int = 1_000,
        delimiter=','
    ):

        edges = list(G.edges(data=True))
        total_edges = len(edges)

        if out == 'kg':
            worker_fn = partial(self._edge_to_kgrow, node_attrs=dict(G.nodes))
            out_path = self.dir_path / Path(self.kg_file_name)
        elif out == 'edge':
            worker_fn = partial(self._edge_to_edgerow, node_attrs=dict(G.nodes))
            out_path = self.dir_path / Path(self.edge_file_name)


        first_chunk = True           # controls header write
        rows_buffer = []             # in‑memory buffer until chunk is full

        with Pool(processes=cpu_count()) as pool:
            with tqdm(total=total_edges, desc=f"Writing {out_path}") as pbar:
                for row in pool.imap_unordered(worker_fn, edges, chunksize=pool_chunksize):
                    rows_buffer.append(row)
                    pbar.update()

                    # When buffer is full → flush to disk
                    if len(rows_buffer) >= chunk_size:
                        df = pd.DataFrame(rows_buffer)
                        df.to_csv(
                            out_path,
                            mode="w" if first_chunk else "a",
                            index=False,
                            header=first_chunk,
                            sep=delimiter, 
                            quoting=1
                        )
                        first_chunk = False
                        rows_buffer.clear()     # free the memory

                # 4. Final tail flush
                if rows_buffer:
                    df = pd.DataFrame(rows_buffer)
                    df.to_csv(
                        out_path,
                        mode="w" if first_chunk else "a",
                        index=False,
                        header=first_chunk,
                        sep=delimiter, 
                        quoting=1
                    )

    def make_graph_output_synaptix(self, opt='01'):
        """
        Different senorios for outputing the data for training
        01: the raw extracted data, converted to a graph (removing duplicate edges) and 
            chainging the node labels to be merged
        02: 01 + filtering the data to have the largest graph outputed
        
        """

        kg = self.adapt_node_labels(self.kg_raw)
        self.make_g(kg)
        if opt == '02':
            self.G = self.get_largest_g()

        self.G = self.sort_nodes(self.G)
        self.out_put_G(self.G)

    def adapt_node_labels(self, kg,
                          mapp_node_type={'TissueLabel': 'CellLabel'}):

        columns = ['x_type', 'y_type']

        for coli in columns:
            print(f'checking values in {coli}')

            for k, v in mapp_node_type.items():
                print(f'Changing valuse of rows with {k} on {coli} column to {v}')
                kg.loc[
                    kg[coli].astype(str).str.contains(k),
                    coli
                ] = v

        return kg
        



if __name__ == '__main__':


    test_opt = "test_opt" in sys.argv[1:]   #True/False #


    data_path = Path(EXTRACT_CONFIG['data_path'])
    opt = {}
    if test_opt:
        data_path= data_path  / Path('test')
        opt = {'nrows':100}

    file_in = data_path / Path(KG_RAW)
    out_dir = data_path / Path(DATA_VER)
    attr_list= ['id', 'type', 'name', 'source', 'uri']
    data = DataPost(out_dir, file_in, attr_list=attr_list, **opt)
    data.make_graph_output_synaptix()

    # opt = {'nrows': 50}

    # opt = {}
    # ver = '02'

    # data_path_synaptix = Path( '../../.images/neo4j/data_primekg/')
    # file_in = data_path_synaptix / Path('kg.csv')
    # out_dir = data_path_synaptix / Path(ver)
    # attr_list= ['id', 'type', 'name', 'source']

    # data = DataPost(out_dir, file_in, attr_list=attr_list, **opt)

    # attr_list= ['id', 'type', 'name', 'source']
    # data.make_g(data.kg_raw)
    # data.out_put_G(data.G)
