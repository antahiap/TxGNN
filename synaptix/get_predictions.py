import json
import os
import pandas as pd
import pickle
from pathlib import Path
import _path
from txgnn import TxEval, TxGNN, TxData


class trained_obj:
    def __init__(self, config, rel_path, mask=False):
        self.config = config

        self.model_path = Path(rel_path) / Path(config['model_path'])
        
        self.mask = mask
        if mask:
            self.model_path = self.model_path / Path(mask)

        self.data_path = config['data_config']['data_path']
        self.nodes =  pd.read_csv(Path(self.data_path) / Path('node.csv'), delimiter='\t')
        self.model = None

    def get_model(self, dump_map=False):

        split_name = self.config["data_config"]["split_name"]
        seed_no = self.config["data_config"]["seed_no"]

        TxDataObj = TxData(data_folder_path=self.data_path)
        TxDataObj.prepare_split(split=split_name, seed=seed_no, no_kg=False)
        self.id_mapping = TxDataObj.retrieve_id_mapping()

        if dump_map:
            dump_file = Path(self.model_path) / 'id_mapping.pkl'
            print(f'dump id_mappaing {dump_file}')
            with open(dump_file, 'wb') as f:
                pickle.dump(self.id_mapping, f)


        TxGNN_obj = TxGNN(
            data =TxDataObj,
            data_map=self.config["data_config"]["data_map"]
            )

        if self.mask and os.path.isfile(self.model_path):
            TxGNN_obj.load_pretrained_graphmask(self.model_path)
        else:
            TxGNN_obj.load_pretrained(self.model_path)
        self.model = TxEval(model = TxGNN_obj)

        return 
    

    def get_predictions(self, id, disease_idx=None, status=None, rel_path=''):
        
        if not disease_idx:
            disease_idx = self.id_mapping['idx2id_disease'].keys()    
            id == 'all'

        print(id)
        save_name = Path(rel_path) / Path(self.model_path) / f'eval_disease_centric_{id}.pkl'
        print('Looking for result in:')
        print(save_name)

        if os.path.exists(save_name):
            print(f'Found!.Read existing predictions.')
            with open(save_name, 'rb') as f:
                result = pickle.load(f)
        elif status:
            print('Read input result')
            print(status)
            with open(status, 'rb') as f:
                result = pickle.load(f)

        else:
            if not self.model:
                self.get_model()
                
            result = self.model.eval_disease_centric(disease_idxs = disease_idx, 
                    relation = 'indication',
                    save_result = True,
                    save_name = save_name)
        return  result

def get_ids(selected_disease, id_map):
    name2id = {v: k for k, v in id_map['id2name_disease'].items()}
    idx = [name2id[name] for name in selected_disease if name in name2id]
    
    id2idx = {v: k for k, v in id_map['idx2id_disease'].items()}
    idx_keys = [id2idx[i] for i in idx if i in id2idx]
    
    return idx_keys

if __name__ == '__main__':

    print("Your message", flush=True)

    log_runs_file = './synaptix/run_log.json'
    # disease_ids = [
    # '5090_13498_8414_10897_33312_10943_11552_14092_12054_11960_11280_11294_11295_11298_11307_11498_12879_13089_13506', 
    # #'12945_11691_12077_8780_11196_27694_11951_10459_11223_11632_14223_12790_12753_13715_7103_13264_13891_14181_4976_5145_11952'
    # ]
    # tag_map = ['node_id', 'node_index']

    with open(log_runs_file, 'r') as f:
        runs = json.load(f)

    id_mapping = runs['011']['model_path']/ Path('id_mapping.pkl')

    with open(id_mapping, 'rb') as f:
        id_mapping = pickle.load(f)

    selected_disease = ['schizophrenia', 'amyotrophic lateral sclerosis']
    disease_idx = get_ids(selected_disease, id_mapping)

    

    m4 = trained_obj(runs['011'], '')
    m4.get_model(dump_map=True)
    m4.get_predictions('01', disease_idx)


    
