import os
import sys
import json
from txgnn import TxData, TxGNN
from constants import DATA_DIR

from datetime import datetime
import log_config 

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

def graph_mask_txgnn(config, id):
    
    split_name = config["data_config"]["split_name"]
    seed_no = config["data_config"]["seed_no"]
    data_path = config["data_config"]["data_path"]

    graphmask_conf = config["graphmask"][id]
    model_path = config["model_path"]
    model_conf = config["model_config"]
    pretrain_conf = config["pretrain"]
    finetune_conf = config["finetune"]


    print(f'Look for data: {data_path}')
    TxDataObj = TxData(data_folder_path = data_path)
    TxDataObj.prepare_split(split = split_name, seed = seed_no, no_kg = False)

    TxGNNObj = TxGNN(
        data=TxDataObj,
        weight_bias_track=False,
        proj_name='TxGNN',
        exp_name='TxGNN', 
        data_map=config["data_config"]["data_map"],
        device=graphmask_conf['device']
    )

    if os.path.exists(model_path):
        TxGNNObj.load_pretrained(model_path)
    else:
        s_time = datetime.now()

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

        config['run_time'] = str((datetime.now() - s_time).total_seconds()/60) + ' min'

        c.update_run_log(config)
        print("End time:", datetime.now())


    
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
    print(sys.stdout) 
    print("Your message", flush=True)
    s_time = datetime.now()
    print("Start time:", s_time)

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
    
    use_log = True
    study_no = '015'
    mask_id = '001'

    # If use_log = True=True, these parameters are not used
    data_name = 'primekg/02' #'primekg' #
    comment = 'remove primekg classes that are not in synaptix'
    n, l, m = 2, 2, 2 #512, 512, 512 #
    np, nf = 2, 500 # 1, 1 #
    bs = 1024 #1024*1000 #
    num_walks = 2 #200
    seed = 4
    data_map = data_map_1

    run_log_file =  'synaptix/run_log.json'
    c = log_config.Config(study_no, run_log_file, use_log)
    config = c.set_config(data_name, data_map, DATA_DIR, n, l, m, np, bs, nf, comment, num_walks, 
                          seed= seed)
    c.update_run_log(config)
    # -------------------------------------------------------------------------------
    # TRAIN GRAPH MASK
    # ----------------------------------    
    config = c.get_log()
    config['graphmask'][mask_id] = c.set_config_mask(device='cuda')
    c.update_run_log(config)
    
    graph_mask_txgnn(config, mask_id)

    config['graphmask'][mask_id]['run_time'] = str((datetime.now() - s_time).total_seconds()/60) + ' min'

    c.update_run_log(config)
    print("End time:", datetime.now())
