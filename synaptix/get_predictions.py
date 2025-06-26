import json
import os
import pandas as pd
import pickle
from pathlib import Path
import _path
from txgnn import TxEval, TxGNN, TxData


class trained_obj:
    def __init__(self, config, rel_path):
        self.config = config

        self.model_path = Path(rel_path) / Path(config['model_path'])
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
            print(f'dump id_mappaing {self.model_path}')
            with open(Path(self.model_path) / 'id_mapping.pkl', 'wb') as f:
                pickle.dump(self.id_mapping, f)


        TxGNN_obj = TxGNN(
            data =TxDataObj,
            data_map=self.config["data_config"]["data_map"]
            )

        TxGNN_obj.load_pretrained(self.model_path)
        self.model = TxEval(model = TxGNN_obj)

        return 
    

    def get_predictions(self, id, disease_idx=None):
        
        if not disease_idx:
            disease_idx = self.id_mapping['idx2id_disease'].keys()    
            id == 'all'

        save_name = Path(self.model_path) / f'eval_disease_centric_{id}.pkl'

        if os.path.exists(save_name):
            print(f'Read existing predictions:')
            print(save_name)
            with open(save_name, 'rb') as f:
                result = pickle.load(f)

        else:
            if not self.model:
                self.get_model()
                
            result = self.model.eval_disease_centric(disease_idxs = disease_idx, 
                    relation = 'indication',
                    save_result = True,
                    save_name = save_name)
        return  result

if __name__ == '__main__':

    log_runs_file = 'model/local_runs/run_log.json'
    # disease_ids = [
    # '5090_13498_8414_10897_33312_10943_11552_14092_12054_11960_11280_11294_11295_11298_11307_11498_12879_13089_13506', 
    # #'12945_11691_12077_8780_11196_27694_11951_10459_11223_11632_14223_12790_12753_13715_7103_13264_13891_14181_4976_5145_11952'
    # ]
    # tag_map = ['node_id', 'node_index']

    
    with open(log_runs_file, 'r') as f:
        runs = json.load(f)

    m4 = trained_obj(runs['002'], '')
    m4.get_model()
    m4.get_predictions()


    
