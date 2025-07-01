
import os
import json

class Config:
    def __init__(self, study_no, run_log_file, use_log):

        self.study_no = study_no
        self.run_log_file = run_log_file
        self.use_log = use_log
        self.runs = {}

    def update_run_log(self, config):

        if os.path.isfile(self.run_log_file):
            with open(self.run_log_file, 'r') as f:
                self.runs = json.load(f)

        self.runs[self.study_no] = config

        with open(self.run_log_file, 'w') as f:
            out = json.dumps(self.runs, indent=2)
            f.write(out)

    def get_log(self):

        with open(self.run_log_file, 'r') as f:
            runs = json.load(f)

        return(runs[self.study_no])

    def set_config(self, data_name, data_map, data_dir, n, l, m, np, bs, nf, comment, num_walks, seed=42):

        if self.use_log:
            config = self.get_log()
            return config

        data_config = {
            "name": data_name,
            "split_name": 'complex_disease',
            "seed_no": seed,
            "data_path": f'{data_dir}/data_{data_name}',
            "data_map": data_map
        }



        model_config = {
            "n_hid": n,
            "n_inp": l,
            "n_out": m,
            "num_walks": num_walks,
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
            "model_path":  f'./model/local_runs/{self.study_no}_{data_name}',
            "data_config": data_config,
            "model_config": model_config,
            "pretrain": pretrain,
            "finetune": finetune,
            "graphmask": {}
        }

        return config

    def set_config_mask(self,
            relation = 'indication',
            learning_rate = 3e-4,
            allowance = 0.005,
            epochs_per_layer = 3,
            penalty_scaling = 1,
            valid_per_n = 20,
            device = 'cpu'
            ):

        config = {
            "relation": relation,
            "learning_rate": learning_rate,
            "allowance": allowance,
            "epochs_per_layer": epochs_per_layer,
            "penalty_scaling": penalty_scaling,
            "valid_per_n": valid_per_n,
            "device": device
        }
        return config