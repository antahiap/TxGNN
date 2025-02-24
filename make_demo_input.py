import numpy as np
import _path
import ast
import json
import pandas as pd
import networkx as nx
import pickle
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sklearn.manifold import TSNE
import os

from tqdm import tqdm

from txgnn import TxData, TxGNN, TxEval
import database as ds

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
        
        self.db = ds.Neo4jApp(
            server='', user=username,  password=password ,          
            database=self.database,
            datapath='TxGNNExplorer_v2')

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

    def disease_drug_edgesource_rank(self):

        def _make_graph(db, disease_id, drug_id):

            data = db.query_attention_pair(disease_id, drug_id)#['paths']
            rows = []

            for item in data['paths']:
                row = {
                    'nodeIds': [x['nodeId'] for x in item['nodes']],
                    'nodeTypes': [x['nodeType'] for x in item['nodes']],
                    'nodeSources': [x['nodeSource'] for x in item['nodes']],
                    'edgeInfos': [x['edgeInfo'] for x in item['edges']],
                    'edgeScore': [x['score'] for x in item['edges']],
                    'avg_score': item['avg_score'],
                    'edgeSource': [x['source'] for x in item['edges']]
                }
                rows.append(row)


            # Create a DataFrame
            paths_df = pd.DataFrame(rows)
            if paths_df.empty:
                return None

            nodesIds = [item for sublist in paths_df['nodeIds'] for item in sublist]
            nodeTypes = [item for sublist in paths_df['nodeTypes'] for item in sublist] 
            nodeSource = [item for sublist in paths_df['nodeSources'] for item in sublist] 

            nodes_set = set(zip(nodesIds, nodeTypes, nodeSource))
            Gp = nx.Graph()
            for ni in nodes_set:
                Gp.add_node(ni[0], type=ni[1], source_name=ni[2])


            # make edge list
            edge_lists = []
            for j, pj in enumerate(paths_df['nodeIds']):
                for i in range(0, len(pj)-1):
                    edge_lists.append([pj[i], pj[i+1], paths_df['edgeScore'][j][i]])
                    wi = float(paths_df['edgeScore'][j][i] )
                    si = paths_df['edgeSource'][j][i]
                    Gp.add_edge(pj[i], pj[i+1], weight=wi, source_name=si)

            return Gp

        def _add_source_nodes_G_w_edge(G, opt):

            node_ids = list(G.nodes())
            query = f"""
                WITH {node_ids} AS id_list
                MATCH (n)-[r:BELONGS_TO]-(s:Source)
                WHERE n.id IN id_list
                RETURN distinct n.id as node_id, s.name as name, s.id as source_id
            """
            source_nodes = self._query_to_dataframe(query)
            source_nodes_info = source_nodes[['name', 'source_id']].drop_duplicates()

            for i, si in source_nodes_info.iterrows():
                G.add_node(si['source_id'], name=si['name'], type='Source')


            df_weight_rows = []
            for i, si in source_nodes.iterrows():
                nid = si['node_id']
                sid = si['source_id']

                nid_edges = list(G.edges(nid, data=True))
                n_s_name = G.nodes[nid]['source_name']

                wi_list = []
                nbr_list = []
                edge_list = []
                edge_list_w = []
                for u, v, prop in nid_edges:
                    if u == nid:
                        nbr_id = v
                    else:
                        nbr_id = u

                    if G.nodes[nbr_id]['type'] == 'Source':
                        continue
                    
                    edge_source = prop['source_name']
                    edge_w = prop['weight']
                    nbr_s_name = G.nodes[nbr_id]['source_name']

                    nbr_list.append(nbr_s_name)
                    edge_list.append(edge_source)
                    edge_list_w.append(edge_w)

                    if nbr_s_name == n_s_name and edge_source == n_s_name:
                        wi = 0.5
                    elif edge_source == n_s_name:
                        wi = 1
                    else:
                        wi = 0
                    wi_list.append(wi)  

                df_weight_rows.append([n_s_name, nbr_list, edge_list, wi_list, edge_list_w])

                if opt == 'general':
                    w = sum(wi_list)
                elif opt == 'score':
                    w = np.dot(np.array(wi_list), np.array(edge_list_w))

                G.add_edge(nid, sid, type='BELONS_TO', weight=w)

            df_weight = pd.DataFrame(df_weight_rows, columns=['N1 source', 'Neighbours', 'Edge sources', 'N1-S weight', 'edge w'])
            return G, df_weight

        def _get_pagerank_by_type(graph, node_type):
        
            pagerank_scores= nx.pagerank(graph, alpha=0.85, max_iter=1000, weight="weight")
            selected_values =  {attr.get("name"): pagerank_scores[n] for n, attr in graph.nodes(data=True) if attr.get("type") == node_type}

            return selected_values

        def _get_diseas_deug_pairs():
            query = '''
                MATCH (d:disease)
                RETURN d.id as disease_id , d.predictions as drugs 
            '''
            return self._query_to_dataframe(query)
        
        def _get_start(d_path, ):
            with open(d_path, 'r') as file:
                result_all = json.load(file)

            last_disease, last_drug = result_all[-1]['disease_drug']

            filtered_df = dd_pairs.loc[(dd_pairs['disease_id'] == last_disease)]
            start_row = filtered_df.index[0]

            return result_all, start_row, last_drug

        def _iter_dd(result_all, sliced_df, last_drug, start_add, out_path):
            j = 0

            for i, row in tqdm(sliced_df.iterrows(), total=len(sliced_df)):
    
                drugs = ast.literal_eval(row['drugs'])
                disease_id = row['disease_id']

                for drug_id, _ in drugs:
                    if start_add:
                        Gi = _make_graph(self.db, disease_id, drug_id)

                        if not Gi:
                            continue
                        
                        G_g = Gi.copy()
                        G_scr = Gi.copy()

                        G_g, _ = _add_source_nodes_G_w_edge(G_g, opt='general')
                        G_scr, _ = _add_source_nodes_G_w_edge(G_scr, opt='score')

                        rank_g = _get_pagerank_by_type(G_g, 'Source')
                        rank_scr = _get_pagerank_by_type(G_scr, 'Source')

                        ri = {
                            "disease_drug": [disease_id, drug_id],
                            "general" : rank_g,
                            "att_score": rank_scr
                        }
                        result_all.append(ri)
                        j +=1

                        if j % 10 == 0:
                            with open(out_path, "w") as f:
                                json.dump(result_all, f, indent=4)
                                print(f"Dumped at iteration {i}")

                        #break
                    if drug_id == last_drug:
                        start_add = True

        dd_pairs = _get_diseas_deug_pairs()
        result_all_path = f'{self.output_dir}/disease_drug_edge_source_ranking.json'

        if os.path.isfile(result_all_path):
            result_all, start_row, last_drug = _get_start(result_all_path)
            start_add = False
        else:
            result_all = []
            start_row = 0
            start_add = True
            last_drug = None

        
        sliced_df = dd_pairs.iloc[start_row:]
        _iter_dd(result_all, sliced_df, last_drug, start_add, result_all_path)

    def close(self):
        """Closes the Neo4j connection."""
        self.driver.close()

        

if __name__ == '__main__':

    
    print("Your message", flush=True)

    data_path = '../../.images/neo4j/data_primekg'
    output_dir = '../Drug_Explorer/drug_server/txgnn_data'
    model_path = 'data/TxGNNExplorer'


    dinp = DemoInput(data_path, output_dir)

    # dinp.make_node_types_json()
    # dinp.make_drug_indication_subset_pkl()
    # dinp.make_edge_types_json()
    # dinp.make_node_name_dict_json()
    # dinp.make_disease_options_json()
    # dinp.make_drug_tsne_json(model_path)
    dinp.disease_drug_edgesource_rank()
    dinp.close()