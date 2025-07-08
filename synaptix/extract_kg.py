import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import _path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

import synaptix_map_edges
import sys

from constants import EXTRACT_CONFIG


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


def getInfo(info, edge_label, config, test_opt):

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

    opt = ''
    if test_opt:
        opt = 'LIMIT 10'

    query = f"""   
        MATCH p=(s{src_tag})-[r:{edges}]-(t{trgt_tag})
        WITH s, t, r, type(r) AS relationship_type, COUNT(r) AS count
        RETURN {text}  COLLECT({{type: relationship_type, count: count}}) AS relationship_types {opt}
    """
    print(query)
    return query_to_dataframe(db_name, driver, query)

def call_all_edges(config, test_opt):
    
    data_path = config['data_path']
    Path(data_path).mkdir(parents=True,exist_ok=True)
    out_name = config['out_name']

    edges_map = pd.DataFrame(columns=['relation'])
    kg_path = data_path / Path(out_name)
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
            df = getInfo(v, k, config, test_opt)
            edges_map = pd.concat([edges_map, df], ignore_index=True)
            edges_map.to_csv(kg_path, sep=",", index=False, quoting=1)
            print('-----------------')

            if test_opt:
                break

    #edges_map.insert(0, 'node_index', range(len(edges_map)))
    edges_map.to_csv(kg_path, sep=",", index=False, quoting=1)
    return edges_map


if __name__ == '__main__':

    print("Your message", flush=True)

    config = EXTRACT_CONFIG

    test_opt = "test_opt" in sys.argv[1:]   #True/False #
    if test_opt:
        config['data_path'] = Path(config['data_path'])  / Path('test')
        print('data_path set to: ')
        print(config['data_path'] )

    map_e = synaptix_map_edges.map_e
    env_path = Path('../../.env')
    load_dotenv(os.path.abspath(env_path))


    driver, db_name,_,_ = get_driver('_SYNAPTIX')
    edges_map = call_all_edges(config, test_opt)
