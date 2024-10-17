
def graph_mask_txgnn(split_name, seed_no, data_path):
    from txgnn import TxData, TxGNN, TxEval
    
    TxData = TxData(data_folder_path = data_path)
    TxData.prepare_split(split = split_name, seed = seed_no, no_kg = False)


    TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN'
              )

    #TxGNN.load_pretrained(f'./models_{split_name}_{seed_no}/')
    TxGNN.load_pretrained('./models/checkpoints_all_seeds/TxGNN_1_complex_disease')

    TxGNN.train_graphmask(relation = 'indication',
                      learning_rate = 3e-4,
                      allowance = 0.005,
                      epochs_per_layer = 3,
                      penalty_scaling = 1,
                      valid_per_n = 20)
    
    output = TxGNN.retrieve_save_gates('.gMask/model_ckpt_GM')
    TxGNN.save_graphmask_model('./gMask/graphmask_model_ckpt_GM')

    return TxGNN

if __name__ == '__main__':

    print("Your message", flush=True)
    TxGNN = graph_mask_txgnn(
        'complex_disease', 
        42, 
        'data/train_data'
        )
    
    from txgnn import TxEval
    TxEval = TxEval(model = TxGNN)
    result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
                                     show_plot = True, 
                                     verbose = True, 
                                     save_result = True,
                                     return_raw = False)
