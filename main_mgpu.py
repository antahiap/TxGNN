import os
import torch
from torch import nn
import sys
import torch.multiprocessing as mp

import tqdm
import time
import torch.distributed as dist

os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

def train_txgnn(TxData, bs, model_save_path):

    from txgnn import TxGNN

    TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN'
              )

    #TxGNN.load_pretrained(f'./models_{split_name}_{seed_no}/')
    TxGNN.model_initialize(n_hid = 100, 
                      n_inp = 100, 
                      n_out = 100, 
                      proto = True,
                      proto_num = 3,
                      attention = False,
                      sim_measure = 'all_nodes_profile',
                      bert_measure = 'disease_name',
                      agg_measure = 'rarity',
                      num_walks = 200,
                      walk_mode = 'bit',
                      path_length = 2)
    
    TxGNN.pretrain(n_epoch = 2, 
               learning_rate = 1e-3,
               batch_size = bs, 
               train_print_per_n = 20)
    
    TxGNN.save_model(model_save_path)

    TxGNN.finetune(n_epoch = 500, 
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20)
    
    TxGNN.save_model(f'{model_save_path}_500_batch')

def run(rank, world_size, devices, dataset, bs=1042):

    from txgnn import TxGNN

    # Set up multiprocessing environment.
    device = devices[rank]
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU training
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )

    # Pin the graph and features in-place to enable GPU access.
    graph = dataset.G.pin_memory_()
    #features = dataset.feature.pin_memory_()
    train_set = dataset.df_train #tasks[0].train_set
    valid_set = dataset.df_valid # tasks[0].validation_set
    #num_classes = dataset.tasks[0].metadata["num_classes"]

    # Create GraphSAGE model. It should be copied onto a GPU as a replica.
    model = TxGNN(dataset)
    model.model_initialize(n_hid = 100, 
                      n_inp = 100, 
                      n_out = 100, 
                      proto = True,
                      proto_num = 3,
                      attention = False,
                      sim_measure = 'all_nodes_profile',
                      bert_measure = 'disease_name',
                      agg_measure = 'rarity',
                      num_walks = 200,
                      walk_mode = 'bit',
                      path_length = 2)
    model = nn.parallel.DistributedDataParallel(model)

    # Model training.
    if rank == 0:
        print("Training...")

    TxGNN.pretrain_mpp(n_epoch = 2, 
               learning_rate = 1e-3,
               batch_size = bs, 
               train_print_per_n = 20)

    # # Test the model.
    # if rank == 0:
    #     print("Testing...")
    # test_set = dataset.tasks[0].test_set
    # test_acc, num_test_items = evaluate(
    #     rank,
    #     model,
    #     graph,
    #     features,
    #     itemset=test_set,
    #     num_classes=num_classes,
    #     device=device,
    # )
    # test_acc = weighted_reduce(test_acc * num_test_items, num_test_items)

    # if rank == 0:
    #     print(f"Test Accuracy {test_acc.item():.4f}")

def main(data_path):
    from txgnn import TxData

    split_name = 'complex_disease'
    seed_no = 42
    bs = 1024
    model_save_path = f'./model/local_runs/models_{split_name}_{seed_no}_batch_{bs}_nhio_half'

    if not torch.cuda.is_available():
        print("No GPU found!")
        return

    devices = [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ][:1]
    world_size = len(devices)

    print(f"Training with {world_size} gpus.")

    # Load and preprocess dataset.
    dataset = TxData(data_folder_path = data_path)
    dataset.prepare_split(split = split_name, seed = seed_no, no_kg = False)
    #dataset = gb.OnDiskDataset(base_dir).load() # gb.BuiltinDataset("ogbn-products").load()

    # Thread limiting to avoid resource competition.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // world_size)

    if world_size > 1:
        # The following launch method is not supported in a notebook.
        mp.set_sharing_strategy("file_system")
        mp.spawn(
            train_txgnn,
            args=(world_size, devices, dataset),
            nprocs=world_size,
            join=True,
        )
    else:
        # train_txgnn(TxData, bs, model_save_path),
        run(0, 1, devices, dataset)


if __name__ == '__main__':

    print("Your message", flush=True)

    data_path = '/home/apakiman/Repo/merck_gds_explr/.images/neo4j/data_primekg'
    print(os.path.exists(data_path))
    main(data_path)
