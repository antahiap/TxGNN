import json
import os
import sys
from datetime import datetime

import pandas as pd
from constants import DATA_DIR
import log_config 

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
    
    use_log = False #True
    study_no = '001'
    
    # If use_log = True=True, these parameters are not used
    data_name = 'primekg'
    comment = 'test'
    n, l, m = 2, 2, 2 #512, 512, 512 #
    np, nf = 1, 1 #2, 500 # 
    bs = 1024*1000 #1024
    

    data_map = data_map_1

    run_log_file =  'synaptix/run_log.json'
    c = log_config.Config(study_no, run_log_file, use_log)
    config = c.set_config(data_name, data_map, DATA_DIR, n, l, m, np, bs, nf, comment)

    c.update_run_log(config)

    # -------------------------------------------------------------------------------
    # TRAIN
    # ----------------------------------    
    train_txgnn(config)

    config['run_time'] = str((datetime.now() - s_time).total_seconds()/60) + ' min'

    c.update_run_log(config)
    print("End time:", datetime.now())