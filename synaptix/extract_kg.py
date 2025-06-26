import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import _path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

import synaptix_map_edges


def query_to_dataframe(database, driver, query):
    with driver.session(database=database) as session:
        result = session.run(query)
        records = [record.data() for record in result]
        return pd.DataFrame(records)
    

def get_driver(STATE):
    uri = os.getenv(f"NEO4J_URI{STATE}")
    username = os.getenv(f"NEO4J_USERNAME{STATE}")
    password = os.getenv(f"NEO4J_PASSWORD{STATE}")
    db_name = os.getenv(f"NEO4J_DB{STATE}")
    print(db_name)
    
    print(STATE, uri, username)
    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver, db_name, username, password


def getInfo(info, edge_label, config):

    filter_node = config['filter_node']
    map_tmplt = config['map_tmplt']

    src = info['source'][0]
    src_tag = f":{src}" if src != '' else src
    src_extrct = '"' + str(src) + '"' if src != '' else "head(labels(s))"

    trgt = info['target'][0]
    trgt_tag = f":{trgt}" if trgt != '' else trgt
    trgt_extrct = '"' + str(trgt) + '"' if trgt != '' else "head(labels(t))"
    
    skip_l = info['skip_list']
    edges = '|'.join(info['edges'])

    src_fltr, trgt_fltr = '', ''
    for kf,vf in filter_node.items():
        if kf in src:
            for kn, vn in vf.items():
                src_fltr += f' AND s.{kn}="{vn}"'
        if kf in trgt:
            for kn, vn in vf.items():
                trgt_fltr += f' AND t.{kn}="{vn}"'


    text = ''
    for k, v in map_tmplt.items():
        if k == 'relation':
            v = '"' + edge_label + '"'
        text += f'{v} AS {k}, '

    query = f"""   
        MATCH p=(s{src_tag})-[r:{edges}]-(t{trgt_tag})
        WITH s, t, r, type(r) AS relationship_type, COUNT(r) AS count
        RETURN {text}  COLLECT({{type: relationship_type, count: count}}) AS relationship_types 
    """
    print(query)
    return query_to_dataframe(db_name, driver, query)

def call_all_edges(config):
    
    data_path_synaptix = config['data_path_synaptix']
    out_name = config['out_name']

    edges_map = pd.DataFrame(columns=['relation'])
    kg_path = data_path_synaptix + out_name
    if os.path.exists(kg_path):
        edges_map = pd.read_csv(kg_path, delimiter=',')

    for k, v in map_e.items():
        print('-----------------')
        if k in edges_map['relation'].unique():
            print(k)
            print('Data already extracted!')
            continue
        if v['status'].startswith('ok'):
            print(k)
            df = getInfo(v, k, config)
            edges_map = pd.concat([edges_map, df], ignore_index=True)
            edges_map.to_csv(kg_path, sep=",", index=False, quoting=1)
            print('-----------------')

    #edges_map.insert(0, 'node_index', range(len(edges_map)))
    edges_map.to_csv(kg_path, sep=",", index=False, quoting=1)
    return edges_map


if __name__ == '__main__':

    print("Your message", flush=True)

    map_e = synaptix_map_edges.map_e
    env_path = Path('../../.env')
    load_dotenv(os.path.abspath(env_path))


    driver, db_name,_,_ = get_driver('_SYNAPTIX')
    config = {
        'data_path_synaptix': '../../.images/neo4j/data_synaptix/',
        'out_name' : 'kg_SYNAPTIX.csv',
        'map_tmplt' : {
                'relation': '',
                'display_relation': 'relationship_type',
                'x_id': 'ID(s)',
                'x_index': 's.uri',
                'x_type': 'head(labels(s))',
                'x_name': 's.prefLabel',
                'x_source': 's.uri', 
                'y_id': 'ID(t)',
                'y_index': 't.uri',
                'y_type': 'head(labels(t))',
                'y_name': 't.prefLabel',
                'y_source': 't.uri', 
                'edge_source': 'r.provenance',
                'relation_synaptix': 'type(r)',
            },

        'filter_node' : {
            'gene': {'species': 'human'}
        }
    }



    edges_map = call_all_edges(config)
