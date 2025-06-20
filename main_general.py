import json
import os
import sys
from datetime import datetime

import pandas as pd
from constants import DATA_DIR, MODEL_PATH

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

def train_txgnn(config):
    from txgnn import TxData, TxGNN, TxEval

    split_name = config["data_config"]["split_name"]
    seed_no = config["data_config"]["seed_no"]
    data_path = config["data_config"]["data_path"]

    model_conf = config["model_config"]
    pretrain_conf = config["pretrain"]
    finetune_conf = config["finetune"]
    model_path = config["model_path"]
    
    print(f'Look for data: {data_path}')
    TxDataObj = TxData(data_folder_path=data_path)
    TxDataObj.prepare_split(split=split_name, seed=seed_no, no_kg=False)


    TxGNNObj = TxGNN(
        data=TxDataObj,
        weight_bias_track=False,
        proj_name='TxGNN',
        exp_name='TxGNN', 
        data_map=config["data_config"]["data_map"]
    )

    if os.path.exists(model_path):
        TxGNNObj.load_pretrained(model_path)
    else:

        TxGNNObj.model_initialize(
            n_hid=model_conf["n_hid"],
            n_inp=model_conf["n_inp"],
            n_out=model_conf["n_out"],
            proto=model_conf["proto"],
            proto_num=model_conf["proto_num"],
            attention=model_conf["attention"],
            sim_measure=model_conf["sim_measure"],
            bert_measure=model_conf["bert_measure"],
            agg_measure=model_conf["agg_measure"],
            num_walks=model_conf["num_walks"],
            walk_mode=model_conf["walk_mode"],
            path_length=model_conf["path_length"],
        )

        TxGNNObj.pretrain(
            n_epoch=pretrain_conf["n_epoch"],
            learning_rate=pretrain_conf["learning_rate"],
            batch_size=pretrain_conf["batch_size"],
            train_print_per_n=pretrain_conf["train_print_per_n"]
        )

        TxGNNObj.save_model(model_path)

    TxGNNObj.finetune(
        n_epoch=finetune_conf["n_epoch"],
        learning_rate=finetune_conf["learning_rate"],
        train_print_per_n=finetune_conf["train_print_per_n"],
        valid_per_n=finetune_conf["valid_per_n"]
    )

    TxGNNObj.save_model(model_path)

def update_run_log(config, run_log_file):

    if os.path.isfile(run_log_file):
        with open(run_log_file, 'r') as f:
            runs = json.load(f)

    else:
        runs = {}

    runs[study_no] = config

    with open(run_log_file, 'w') as f:
        out = json.dumps(runs, indent=2)
        f.write(out)

def get_log(run_log_file, study_no):

    with open(run_log_file, 'r') as f:
        runs = json.load(f)

    return(runs[study_no])

def set_config():

    if use_log:
        config = get_log(run_log_file, study_no)
        return config
    
    data_config = {
        "name": data_name,
        "split_name": 'complex_disease',
        "seed_no": 42,
        "data_path": f'{DATA_DIR}/data_{data_name}',
        "data_map": data_map
    }



    model_config = {
        "n_hid": n,
        "n_inp": l,
        "n_out": m,
        "num_walks": 2,
        "path_length": 2,
        "proto": True,
        "proto_num": 3,
        "attention": False,
        "sim_measure": "all_nodes_profile",
        "bert_measure": "disease_name",
        "agg_measure": "rarity",
        "walk_mode": "bit"
    }

    pretrain = {
        "n_epoch": np,
        "batch_size": bs,
        "learning_rate": 1e-3,
        "train_print_per_n": 20
    }

    finetune = {
        "n_epoch": nf,
        "learning_rate": 5e-4,
        "train_print_per_n": 5,
        "valid_per_n": 20
    }


    config = {
        "comment": comment, 
        "model_path":  f'./model/local_runs/{data_name}_{study_no}',
        "data_config": data_config,
        "model_config": model_config,
        "pretrain": pretrain,
        "finetune": finetune,
    }

    return config


if __name__ == '__main__':

    print("Your message", flush=True)
    s_time = datetime.now()
    print("Start time:", s_time)

    '''
    from paper Appendix:

    S4.4 Hyperparameter tuning

    We conduct hyperparameter tuning using Hyperband on validation set micro AUROC using complex
    disease split following two stages. The ﬁrst is to optimize the parameters for pre-training and ﬁx ﬁne-
    tuning parameters, where we conduct a sweep of grid search with a learning rate of {1e−4, 5e−4, 1e−
    3}, batch size of {1024, 2048}, and epoch size of {1, 2, 3}. Next, we ﬁx the pre-training parameters
    and do a grid search for ﬁne-tuning parameters with the hidden size of {64, 128, 256, 512}, input
    size of {64, 128, 256, 512}, output size of {64, 128, 256, 512}, number of inter-disease prototypes of
    {3, 5, 10, 20, 50} and learning rate of {1e − 4, 5e − 4, 1e − 3}. 
    
    We obtain a ﬁnal set of hyperparameters
    with a pre-training learning rate of 1e − 3, batch size of 1024, epoch size of 2, ﬁne-tuning learning rate
    of 5e − 4, hidden size of 512, input size of 512, output size of 512, number of prototypes 3.
    
    '''

    data_map_1 = {
            "disease_etypes_all": ['disease_disease', 'disease_phenotype_positive', 'rev_exposure_disease', 'rev_disease_protein'],
            "disease_nodes_all": ['disease', 'effect/phenotype', 'exposure', 'gene/protein'],
            "disease_etypes": ['disease_disease', 'rev_disease_protein'],
            "disease_nodes": ['disease', 'gene/protein'] 
        }
    data_map_2 = {
            "disease_etypes_all": ['disease_disease', 'disease_phenotype_positive', 'rev_exposure_disease'],#, 'rev_disease_protein'],
            "disease_nodes_all": ['disease', 'effect/phenotype', 'exposure'], #, 'gene/protein'],
            "disease_etypes": ['disease_disease'],#, 'rev_disease_protein'],
            "disease_nodes": ['disease']#, 'gene/protein'] 
        }
    data_map_3 = {
            "disease_etypes_all": ['disease_disease', 'disease_phenotype_positive', 'rev_exposure_disease', 'disease_protein'],
            "disease_nodes_all": ['disease', 'effect/phenotype', 'exposure', 'target'],
            "disease_etypes": ['disease_disease', 'disease_protein'],
            "disease_nodes": ['disease', 'target'] 
        }
    # -------------------------------------------------------------------------------
    # Set study parameter
    # ----------------------------------    
    
    use_log = False
    study_no = '005'
    
    # If use_log = True=True, these parameters are not used
    data_name = 'synaptix' #'primekg'
    comment = 'using target for gene/protein for disease, and disease_protein edge '
    n, l, m = 512, 512, 512 #2, 2, 2
    np, nf = 2, 500 # 1, 1
    bs = 1024# *1000
    

    data_map = data_map_3

    run_log_file =  f'{MODEL_PATH}/run_log.json'
    config = set_config()

    update_run_log(config, run_log_file)

    # -------------------------------------------------------------------------------
    # TRAIN
    # ----------------------------------    
    train_txgnn(config)

    config['run_time'] = str((datetime.now() - s_time).total_seconds()/60) + ' min'

    update_run_log(config, run_log_file)
    print("End time:", datetime.now())