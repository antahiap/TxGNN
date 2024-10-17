
def train_txgnn(split_name, seed_no, data_path):
    from txgnn import TxData, TxGNN, TxEval
    
    TxData = TxData(data_folder_path = data_path)
    TxData.prepare_split(split = split_name, seed = seed_no, no_kg = False)


    TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN'
              )

    TxGNN.load_pretrained(f'./models_{split_name}_{seed_no}/')
    # TxGNN.model_initialize(n_hid = 100, 
    #                   n_inp = 100, 
    #                   n_out = 100, 
    #                   proto = True,
    #                   proto_num = 3,
    #                   attention = False,
    #                   sim_measure = 'all_nodes_profile',
    #                   bert_measure = 'disease_name',
    #                   agg_measure = 'rarity',
    #                   num_walks = 200,
    #                   walk_mode = 'bit',
    #                   path_length = 2)
    
    # TxGNN.pretrain(n_epoch = 2, 
    #            learning_rate = 1e-3,
    #            batch_size = 1024, 
    #            train_print_per_n = 20)

    TxGNN.finetune(n_epoch = 500, 
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20)
    
    TxGNN.save_model(f'./models_{split_name}_{seed_no}_500')

if __name__ == '__main__':

    print("Your message", flush=True)
    train_txgnn(
        'complex_disease', 
        42, 
        'data/train_data'
        )