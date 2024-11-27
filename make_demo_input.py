import json
import pandas as pd
import pickle

from txgnn import TxData, TxGNN, TxEval

class DemoInput():

    def __init__(self, data_path, output_dir):

        # self.Tx_data = TxData(data_folder_path = data_path)
        # self.Tx_data.prepare_split(split = 'complex_disease', seed = 42, no_kg = False)
        
        # self.Tx_GNN = TxGNN(data = self.Tx_data, 
        #           weight_bias_track = False,
        #           proj_name = 'TxGNN',
        #           exp_name = 'TxGNN',
        #           )
        self.output_dir = output_dir

        self.df_kg = pd.read_csv(f'{data_path}/kg_directed.csv')
        self.df_kg0 = pd.read_csv(f'{data_path}/kg.csv')
        self.df_edges = pd.read_csv(f'{data_path}/edges.csv')

    def make_node_types_json(self):
        df = self.df_kg
        node_types = sorted(list(df['x_type'].unique()))

        with open(f'{self.output_dir}/node_types.json', 'w') as file:
            json.dump(node_types, file,  indent=4)

    def make_drug_indication_subset_pkl(self):
        
        df = self.df_kg
        drug_with_indications = list(df[df['relation']=='indication']['x_id'].unique()) 

        with open(f'{self.output_dir}/drug_with_indication_subset.pkl', 'wb') as file:
            pickle.dump(drug_with_indications, file)

    def make_edge_types_json(self):

        edge_info = self.df_edges.drop_duplicates(keep='first', subset='relation')
        edges_only = self.df_kg.drop_duplicates(keep='first', subset='relation').sort_values(by='relation')

        output = {}
        for _, row in edges_only.iterrows():
            key = row["relation"]
            if key not in output:
                edge_info_value = edge_info[
                     edge_info['relation'] == key]["display_relation"].to_list()[0]

                output[key] = {
                    "nodes": [row["x_type"], row["y_type"]],
                    "edgeInfo": edge_info_value,
                }


        with open(f'{output_dir}/edge_types.json', 'w') as file:
            json.dump(output, file,  indent=4)

    def make_node_name_dict_json(self):

        node_types = sorted(list(self.df_kg['x_type'].unique()))
        output = {}
        for n in node_types:
            n_list = {}
            df_filter = self.df_kg0[self.df_kg0['x_type'] == n].drop_duplicates(subset='x_id', keep='first')

            # if n != 'disease':
            #     continue
            for _, row in df_filter.iterrows():
                if n in  ['disease', 'drug' ]:
                    key = row["x_id"]
                else:
                    try:
                        key = str(float(row["x_id"]))
                    except ValueError:
                        key = row["x_id"]


                if key not in n_list:
                
                    n_list[key] = row['x_name']
            output[n] = n_list

        with open(f'{output_dir}/node_name_dict.json', 'w') as file:
            json.dump(output, file)#,  indent=4)

if __name__ == '__main__':

    data_path = '../../.images/neo4j/data_primekg'
    output_dir = '../Drug_Explorer/drug_server/txgnn_data'


    dinp = DemoInput(data_path, output_dir)

    # dinp.make_node_types_json()
    # dinp.make_drug_indication_subset_pkl()
    # dinp.make_edge_types_json()
    dinp.make_node_name_dict_json()