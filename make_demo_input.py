import json
import pandas as pd
import pickle
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sklearn.manifold import TSNE
import os

from txgnn import TxData, TxGNN, TxEval

class DemoInput():

    def __init__(self, data_path, output_dir, env_path="../../.env"):

        self.output_dir = output_dir
        self.Tx_GNN = None

        # self.df_kg = pd.read_csv(f'{data_path}/kg_directed.csv')
        # self.df_kg0 = pd.read_csv(f'{data_path}/kg.csv')
        # self.df_edges = pd.read_csv(f'{data_path}/edges.csv')

        load_dotenv(os.path.abspath(env_path))
        uri = os.getenv("NEO4J_URI_TXGNN")
        username = os.getenv("NEO4J_USERNAME_TXGNN")
        password = os.getenv("NEO4J_PASSWORD_TXGNN")
        self.database = os.getenv("NEO4J_DB_TXGNN") 
        

        self.driver = GraphDatabase.driver(uri, auth=(username, password))


    def _query_to_dataframe(self, query):
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            records = [record.data() for record in result]
            return pd.DataFrame(records)
        
    def _load_txgnn(self):

        self.Tx_data = TxData(data_folder_path = data_path)
        self.Tx_data.prepare_split(split = 'complex_disease', seed = 42, no_kg = False)
        
        self.Tx_GNN = TxGNN(data = self.Tx_data, 
                  weight_bias_track = False,
                  proj_name = 'TxGNN',
                  exp_name = 'TxGNN',
                  )

    def make_node_name_dict_json(self):

        def _df_to_json(df):
            
            node_types = sorted(list(df['label'].unique()))
            output = {}
            
            for n in node_types:
                n_list = {}
                df_filter = df[df['label'] == n]

                # if n != 'disease':
                #     continue
                for _, row in df_filter.iterrows():
                    if n in  ['disease', 'drug' ]:
                        key = row["id"]
                    else:
                        try:
                            key = str(float(row["id"]))
                        except ValueError:
                            key = row["id"]


                    if key not in n_list:
                    
                        n_list[key] = row['name']
                output[n] = n_list

            with open(f'{output_dir}/node_name_dict.json', 'w') as file:
                json.dump(output, file)#,  indent=4)

        query = """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, n.name AS name, n.id AS id
            ORDER BY label
            """
        df = self._query_to_dataframe(query)
        _df_to_json(df)

    def make_disease_options_json(self):
        
        def _df_to_json(df):

            output = list(zip(df['id'], df['treatable']))

            with open(f'{output_dir}/disease_options.json', 'w') as file:
                json.dump(output, file)#,  indent=4)


        query = """
            MATCH (d:disease)
            OPTIONAL MATCH (d)-[:rev_indication]->(dr:drug)
            RETURN DISTINCT d.id AS id, 
                   CASE 
                       WHEN dr IS NOT NULL THEN 'true' 
                       ELSE 'false' 
                   END AS treatable
            ORDER BY d.id
            """
        
        df = self._query_to_dataframe(query)
        _df_to_json(df)

    def make_node_types_json(self):

        query = """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN distinct label
            ORDER BY label
            """
        df = self._query_to_dataframe(query)
        node_types = list(df['label'])


        with open(f'{self.output_dir}/node_types.json', 'w') as file:
            json.dump(node_types, file,  indent=4)

    def make_drug_indication_subset_pkl(self):        

        query = """
            MATCH (d:drug)-[r]-() 
            WHERE TYPE(r)='indication' 
            RETURN DISTINCT d.id as id 
            ORDER BY id
            """
        df = self._query_to_dataframe(query)
        drug_with_indications = list(df) 

        with open(f'{self.output_dir}/drug_with_indication_subset.pkl', 'wb') as file:
            pickle.dump(drug_with_indications, file)

    def make_edge_types_json(self):

        def _df_to_json(df):

            output = {}
            for _, row in df.iterrows():
                key = row["type"]
                if key not in output:

                    output[key] = {
                        "nodes": [row["src"], row["trgt"]],
                        "edgeInfo":  row['type'],
                    }

            with open(f'{output_dir}/edge_types.json', 'w') as file:
                json.dump(output, file,  indent=4)

        query = """
            MATCH (n)-[r]-(m)
            UNWIND labels(n) AS src 
            UNWIND labels(n) AS trgt
            RETURN distinct type(r) as type, src, trgt 

            """
        #, r.info as info
        df = self._query_to_dataframe(query)
        _df_to_json(df)

    def make_drug_tsne_json(self, model_path):
            
        if not self.Tx_GNN:
            self._load_txgnn()

        def _get_tsne_embedding():
            
            self.Tx_GNN.load_pretrained(model_path)
            embed_data = self.Tx_GNN.retrieve_embedding()
            drug_data = embed_data['drug']

            tsne = TSNE(n_components=2, perplexity=100, random_state=42)
            drug_tsne_results = tsne.fit_transform(drug_data)
            drug_tsne = drug_tsne_results.tolist()

            return drug_tsne
        
        drug_tsne = _get_tsne_embedding()
        # -------------------------------------------------------------
        # Currently neo4j id is not matching the mapping
        # query = """   
        #     MATCH (d:drug)
        #     RETURN d.id as id ORDER BY ID(d)
        #     """
        # df = self._query_to_dataframe(query)
        # ids = list(df['id'])

        #  the zip doesn't keep all the data
        # output_tsne = dict(zip(ids, drug_tsne))   
        # -------------------------------------------------------------

        idx2id_drug = self.Tx_data.retrieve_id_mapping()['idx2id_drug']
        idx2id_drug_sorted = dict(sorted(idx2id_drug.items(), key=lambda item: item[0]))
        
        output_tsne = {}
        for i, di in enumerate(drug_tsne):
            drug_name = idx2id_drug_sorted[i]
            output_tsne[drug_name] = di
        
        with open(f'{output_dir}/drug_tsne.json', 'w') as file:
            json.dump(output_tsne, file)

    def close(self):
        """Closes the Neo4j connection."""
        self.driver.close()



if __name__ == '__main__':

    data_path = '../../.images/neo4j/data_primekg'
    output_dir = '../Drug_Explorer/drug_server/txgnn_data'
    model_path = 'data/TxGNNExplorer'


    dinp = DemoInput(data_path, output_dir)

    # dinp.make_node_types_json()
    # dinp.make_drug_indication_subset_pkl()
    # dinp.make_edge_types_json()
    # dinp.make_node_name_dict_json()
    # dinp.make_disease_options_json()
    dinp.make_drug_tsne_json(model_path)
    dinp.close()