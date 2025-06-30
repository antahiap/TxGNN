import sys
import json
from txgnn import TxData, TxGNN

from datetime import datetime
from constants import DATA_DIR, MODEL_PATH
from main_general import get_log, update_run_log

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

def graph_mask_txgnn(config, id):
    
    split_name = config["data_config"]["split_name"]
    seed_no = config["data_config"]["seed_no"]
    data_path = config["data_config"]["data_path"]

    graphmask_conf = config["graphmask"][id]
    model_path = config["model_path"]


    print(f'Look for data: {data_path}')
    TxDataObj = TxData(data_folder_path = data_path)
    TxDataObj.prepare_split(split = split_name, seed = seed_no, no_kg = False)

    TxGNNObj = TxGNN(
        data=TxDataObj,
        weight_bias_track=False,
        proj_name='TxGNN',
        exp_name='TxGNN', 
        data_map=config["data_config"]["data_map"],
        device='cpu'
    )

    TxGNNObj.load_pretrained(model_path)
    
    TxGNNObj.train_graphmask(
                relation = graphmask_conf['relation'], 
                learning_rate = graphmask_conf['learning_rate'], 
                allowance = graphmask_conf['allowance'], 
                epochs_per_layer = graphmask_conf['epochs_per_layer'], 
                penalty_scaling = graphmask_conf['penalty_scaling'], 
                valid_per_n = graphmask_conf['valid_per_n'], 
                )
    
    
    TxGNNObj.save_graphmask_model(model_path + f'/{id}')

    return TxGNN


if __name__ == '__main__':

    print("Your message", flush=True)
    s_time = datetime.now()
    print("Start time:", s_time)


    # -------------------------------------------------------------------------------
    # Set study parameter
    # ----------------------------------    
    
    use_log = True
    study_no = '001'
    mask_id = '001'

    run_log_file =  f'{MODEL_PATH}/run_log.json'
    config = get_log()

    # -------------------------------------------------------------------------------
    # TRAIN GRAPH MASK
    # ----------------------------------    
    graph_mask_txgnn(config, mask_id)

    config['run_time_mask'] = str((datetime.now() - s_time).total_seconds()/60) + ' min'

    update_run_log(config, run_log_file)
    print("End time:", datetime.now())
