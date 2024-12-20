import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

def graph_mask_txgnn(split_name, seed_no, data_path, tag):
    from txgnn import TxData, TxGNN, TxEval
    
    TxData = TxData(data_folder_path = data_path)
    TxData.prepare_split(split = split_name, seed = seed_no, no_kg = False)


    TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN',
              #device='cpu'
              )

    #TxGNN.load_pretrained(f'./models_{split_name}_{seed_no}/')
    model_name = f'models_{split_name}_{seed_no}_{tag}' 
    TxGNN.load_pretrained(f'./models/local_runs/{model_name}')
    

    TxGNN.train_graphmask(relation = 'indication',
                  learning_rate = 3e-4,
                  allowance = 0.005,
                  epochs_per_layer = 3,
                  penalty_scaling = 1,
                  valid_per_n = 20)
    
    
    #output = TxGNN.retrieve_save_gates('.gMask/model_ckpt_GM')
    TxGNN.save_graphmask_model(f'./models/gMask/graphmask_{model_name}')

    return TxGNN

if __name__ == '__main__':

    print("Your message", flush=True)
    TxGNN = graph_mask_txgnn(
        'complex_disease', 
        42, 
        'data/train_data', 
        tag = '500_batch_1024'#_nhio_half'
        )
    
    from txgnn import TxEval
    TxEval = TxEval(model = TxGNN)
    result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
                                     show_plot = True, 
                                     verbose = True, 
                                     save_result = True,
                                     return_raw = False)
