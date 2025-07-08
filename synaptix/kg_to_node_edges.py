import os
from pathlib import Path
import networkx as nx
import pandas as pd
import json

class DataPost:
    def __init__(self, dir_path, file_in, 
                 delimiter=',',
                 node_file_name = 'node.csv',
                 edge_file_name = 'edges.csv',
                 kg_file_name = 'kg.csv',
                 nrows=None
                 ): 
        
        self.dir_path = Path (dir_path)
        if not os.path.isdir(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        self.file_in = file_in 
        if file_in:
            if nrows:
                self.kg_raw = pd.read_csv(self.file_in, delimiter=delimiter, nrows=nrows)
            else:
                self.kg_raw = pd.read_csv(self.file_in, delimiter=delimiter)
        else:
            self.kg = None

        self.nodes = None
        self.edges = None
        self.G = None
        self.kg = None

        self.node_file_name = node_file_name
        self.edge_file_name = edge_file_name
        self.kg_file_name = kg_file_name

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

    def write_edges(self, delimiter=','):
        print('Write edges.csv ... ')
        out_file = self.dir_path / Path(self.edge_file_name)
        self.edges.to_csv(out_file, sep=delimiter, index=False)

    def write_kg(self, delimiter=','):
        print('Write kg.csv ...')
        out_file = self.dir_path / Path(self.kg_file_name)
        self.kg.to_csv(out_file, sep=delimiter, index=False)

    def make_g(self, kg,
               col_list=['id', 'type', 'name', 'source', 'uri']
               ):
        print('Load KG to a graph')
        
        def add_nodes_from_df(df, node_prefixes, col_list):
            G = nx.MultiDiGraph()
            for prefix in node_prefixes:
                cols = [f'{prefix}_{col}' for col in col_list]
                attr_names = dict(zip(cols, col_list))
                node_df = df.drop_duplicates(subset=f'{prefix}_index')

                for _, row in node_df.iterrows():
                    node_index = row[f'{prefix}_index']
                    attributes = {attr_names[col]: row[col] for col in cols if col in row}

                    if G.has_node(node_index):
                        existing_type = G.nodes[node_index].get('type')
                        new_type = row.get(f'{prefix}_type')

                        if existing_type != new_type:
                            raise ValueError(
                                f"[Error] Node ID '{node_index}' already exists with type '{existing_type}', "
                                f"but new type is '{new_type}'. Type mismatch detected."
                            )
                        continue  # Node already exists with same type, skip adding again

                    
                    G.add_node(node_index, **attributes)

            return G


        self.G = add_nodes_from_df(kg, node_prefixes=['x', 'y'], col_list=col_list)

        # Add edges
        self.G.add_edges_from(
            zip(
                kg['x_index'], 
                kg['y_index'], 
                kg[['relation', 'display_relation']].to_dict('records')
            )
        )

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
        #G_renumbered =  nx.convert_node_labels_to_integers(G_renumbered_0, first_label=0, ordering='default')
    
        # neo4j_ids_mapping = dict(zip(G_renumbered.nodes(), G_renumbered_0.nodes()))

        # with open(self.dir_path + neo4j_map_out, 'w') as f:
        #     json.dump(neo4j_ids_mapping, f, indent=2)
        return G_renumbered_0

    def out_put_G(self, G ,
                node_attr_list= ['id', 'type', 'name', 'source', 'uri'],
                kg_attr_list = ['id', 'type', 'name', 'source', 'uri']
    ):

        self.out_nodes(G, node_attr_list)
        self.out_edges(G)
        self.out_kg(G, kg_attr_list)

        self.write_nodes()
        self.write_edges()
        self.write_kg()

    def out_nodes(self, G, 
                  attr_list= ['id', 'type', 'name', 'source', 'uri'],
                  prefix='node'):
        print('Process nodes ... ')
        node_rows = []
        for node_index, attrs in G.nodes(data=True):
            row = {f'{prefix}_index': node_index}
            for attr in attr_list:
                row[f'{prefix}_{attr}'] = attrs.get(attr, '')
            
            node_rows.append(row)

        self.nodes = pd.DataFrame(node_rows)

    def out_edges(self, G):
        # Efficient extraction using list comprehension for edges

        print('Process edges ... ')
        edge_rows = []
        for u, v, data in G.edges(data=True):
            x = G.nodes[u]
            y = G.nodes[v]

            row = {
                'relation': data.get('relation', ''),
                'display_relation': data.get('display_relation', ''),
                'x_index':u,  # Use node ID for x_index
                'y_index':v  # Use node ID for y_index
            }
            edge_rows.append(row)
        self.edges = pd.DataFrame(edge_rows)
        
    def out_kg(self, G, 
                attr_list = ['id', 'type', 'name', 'source', 'uri']):

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
            for attr in attr_list:
                row[f'x_{attr}'] = x.get(attr, '')
            
            for attr in attr_list:
                row[f'y_{attr}'] = y.get(attr, '')

            rows.append(row)

        self.kg = pd.DataFrame(rows)

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

    data_path_synaptix = Path( '../../.images/neo4j/data_synaptix/')
    file_in = data_path_synaptix / Path('kg_mapped_manual.csv')
    out_dir = data_path_synaptix / Path('')
    data = DataPost(out_dir, file_in)#, nrows=100)
    data.make_graph_output_synaptix()


    # data_path_synaptix = Path( '../../.images/neo4j/data_primekg/')
    # file_in = data_path_synaptix / Path('kg.csv')
    # out_dir = data_path_synaptix / Path('02')
    # data = DataPost(out_dir, file_in)#, nrows=10)# 000)


    # pkg = data.kg_raw
    # attr_list= ['id', 'index', 'type', 'name', 'source']
    # data.make_g(pkg, col_list=attr_list)
    # # data.make_g(pkg_filtered, col_list=attr_list)
    # data.out_kg(data.G)
    # data.write_kg()