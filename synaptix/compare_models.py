from pathlib import Path
import pickle
import json
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

import _path
from synaptix import get_predictions as gp

class CompareModel:
    def __init__(self, runs, rel_path= ''):

        self.rel_path = Path(rel_path)
        self.runs = runs

        self.model_path = None
        self.result_file = None
        self.results = {}
        self.flatten_result = []

    def load_run(self, config):

        self.model_path = config['model_path']

        result_exists = None
        if not self.model_path == "":
            model_path =  Path(self.model_path)
            id_mapping =  self.rel_path / model_path/ Path('id_mapping.pkl')
        else:
            self.result_file = config['result_file']
            id_mapping = self.rel_path / Path(self.result_file)
            result_exists = self.rel_path / id_mapping

        with open(id_mapping, 'rb') as f:
            id_mapping = pickle.load(f)

        return id_mapping, result_exists

    def load_results(self, id, c_config):
        for i, ri in enumerate(c_config['runs']):

            run = self.runs[ri]
            m = gp.trained_obj(run, self.rel_path)

            try:
                id_map, result_exists = self.load_run(run)
            except FileNotFoundError:
                m.get_model(dump_map=True)
                id_map, result_exists = self.load_run(run)

            selected_disease = c_config['selected_disease'][i]
            disease_idx = self.get_ids(selected_disease, id_map)

            self.results[ri] = m.get_predictions( id, disease_idx=disease_idx, status=result_exists)
        return 'Done!' 

    def get_ids(self, selected_disease, id_map):

        name2id = {v: k for k, v in id_map['id2name_disease'].items()}
        idx = [name2id[name] for name in selected_disease if name in name2id]

        id2idx = {v: k for k, v in id_map['idx2id_disease'].items()}
        idx_keys = [id2idx[i] for i in idx if i in id2idx]

        return idx_keys

    def get_drug_names(self, drug_list, id_map):
        drugs =  [v for k, v in id_map['id2name_drug'].items() if k in drug_list]
        return drugs #[k for k, v in id_map['idx2id_disease'].items() if v in idx]

    def get_disease_name(self, disease, id_map):
        return id_map['id2name_disease'][disease]

    def sort_predictions(self, rid, nrows=3):

        predictions = {}

        result_i = self.results[rid]
        id_map, _ = self.load_run(self.runs[rid])

        for k, drug_list in result_i['Prediction'].items():
            disease_name = self.get_disease_name(k, id_map)

            df = pd.DataFrame.from_dict(drug_list, orient='index', columns=['pred'])
            df.reset_index(inplace=True)
            df.columns = ['drug_id', 'pred']
            selected_drugs = df.sort_values(by='pred', ascending=False)[:nrows]
            drug_ids = selected_drugs['drug_id'].to_list()

            drug_names = self.get_drug_names(drug_ids, id_map)
            drug_pred = selected_drugs['pred'].to_list()
            predictions[disease_name] = (drug_names, drug_pred)
        return predictions

    def flatten_scores(self, model_id, nrows):
        print((model_id, nrows))
        result_dict = self.sort_predictions(model_id, nrows)

        rows = []
        for disease, (drugs, scores) in result_dict.items():

            d_tag = disease.split(' ')[0]
            row = {'Disease': disease, 'Model': model_id}
            for i, score in enumerate(scores):
                row = {
                    'Drug': drugs[i][:15],
                    'Prediction': score,
                    'Model': model_id,
                    'Disease': d_tag
                }
                rows.append(row)
        self.flatten_result = rows
        return rows

    def make_corr_plot(self, compare):

        comp = []
        for ri in compare:
            comp += self.flatten_scores(ri, -1)
        df = pd.DataFrame(comp)

        df_pivot = df.pivot_table(
            index=["Drug", "Disease"], 
            columns="Model", 
            values="Prediction"
        ).reset_index()


        df_valid = df_pivot.dropna()

        # Calculate correlation
        correlation = df_valid[compare].corr().iloc[0,1]

        g = sns.lmplot(
            data=df_valid,
            x=compare[0],
            y=compare[1],
            col='Disease',
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"},
            height=4, aspect=1.2
        )

        g.set_axis_labels(f"Model {compare[0]} Prediction", f"Model {compare[1]} Prediction")

        # Add a shared title across all subplots
        g.fig.suptitle(
            f"Correlation between Model {compare[0]} and Model {compare[1]} Predictions\nr = {correlation:.3f}",
            fontsize=12,
            y=1.05  # adjust position above facets
        )

        g.set_titles(col_template="{col_name}")  # clean facet titles
        g.tight_layout()
        for ax in g.axes.flatten():
            ax.grid(True)
        plt.show()

if __name__ == '__main__':

    log_runs_file = './synaptix/run_log.json'
    with open(log_runs_file, 'r') as f:
        runs = json.load(f)


    compare_log = {
        '01':{
            'runs': ['002', '004', '003', '005', '100', '006'],
            'selected_disease':[
                 ['schizophrenia', 'amyotrophic lateral sclerosis'],
                 ['schizophrenia 4', 'amyotrophic lateral sclerosis'], 
                 ['schizophrenia', 'amyotrophic lateral sclerosis'],
                 ['schizophrenia 4', 'amyotrophic lateral sclerosis'], 
                 ['schizophrenia', 'amyotrophic lateral sclerosis'],
                 ['schizophrenia 4', 'amyotrophic lateral sclerosis'], 
            ]
        }, 
    }


    id = '01'
    c_config = compare_log[id]

    comp = CompareModel(runs)
    comp.load_results(id, c_config) 
    comp.make_corr_plot(['002', '003'])



