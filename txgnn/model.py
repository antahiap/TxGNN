import dgl
from dgl.ops import edge_softmax
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils import data
import pandas as pd
import copy
import os
import random

import warnings
warnings.filterwarnings("ignore")
from .utils import sim_matrix, exponential, obtain_disease_profile, obtain_protein_random_walk_profile, convert2str
from .graphmask.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from .graphmask.squeezer import Squeezer
from .graphmask.sigmoid_penalty import SoftConcrete

class DistMultPredictor(nn.Module):
    def __init__(self, n_hid, w_rels, G, rel2idx, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device, data_map):
        super().__init__()
        
        self.proto = proto
        self.sim_measure = sim_measure
        self.bert_measure = bert_measure
        self.agg_measure = agg_measure
        self.num_walks = num_walks
        self.walk_mode = walk_mode
        self.path_length = path_length
        self.exp_lambda = exp_lambda
        self.device = device
        self.W = w_rels
        self.rel2idx = rel2idx
        
        self.etypes_dd =  [tuple(t) for t in data_map['dd_etypes']]  #[('drug', 'contraindication', 'disease'), 
                        #    ('drug', 'indication', 'disease'),
                        #    ('drug', 'off-label use', 'disease'),
                        #    ('disease', 'rev_contraindication', 'drug'), 
                        #    ('disease', 'rev_indication', 'drug'),
                        #    ('disease', 'rev_off-label use', 'drug')]
        
        self.node_types_dd = ['disease', 'drug']
        
        if proto:
            self.W_gate = {}
            for i in self.node_types_dd:
                temp_w = nn.Linear(n_hid * 2, 1)
                nn.init.xavier_uniform_(temp_w.weight)
                self.W_gate[i] = temp_w.to(self.device)
            self.k = proto_num
            self.m = nn.Sigmoid()
                   
            if sim_measure in ['bert', 'profile+bert']:
                
                data_path = os.path.join(data_folder, 'kg.csv')
                        
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                                
                self.disease_dict = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].values)
                self.disease_dict.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].values))
                
                if bert_measure == 'disease_name':
                    self.bert_embed = np.load('/n/scratch3/users/k/kh278/bert_basic.npy')
                    df_nodes_bert = pd.read_csv('/n/scratch3/users/k/kh278/nodes.csv')
                    
                elif bert_measure == 'v1':
                    self.bert_embed = np.load('/n/scratch3/users/k/kh278/disease_embeds_single_def.npy')
                    df_nodes_bert = pd.read_csv('/n/scratch3/users/k/kh278/disease_nodes_for_BERT_embeds.csv')
                
                df_nodes_bert['node_id'] = df_nodes_bert.node_id.apply(lambda x: convert2str(x))
                self.id2bertindex = dict(zip(df_nodes_bert.node_id.values, df_nodes_bert.index.values))
                
            self.diseases_profile = {}
            self.sim_all_etypes = {}
            self.diseaseid2id_etypes = {}
            self.diseases_profile_etypes = {}
            
            self.disease_etypes_all = data_map['disease_etypes_all'] #['disease_disease', 'disease_phenotype_positive', 'rev_exposure_disease'] #, 'rev_disease_protein'
            self.disease_nodes_all = data_map['disease_nodes_all'] #['disease', 'effect/phenotype', 'exposure'] #, 'gene/protein'
            
            self.disease_etypes = data_map['disease_etypes'] #['disease_disease'] #, 'rev_disease_protein']
            self.disease_nodes = data_map['disease_nodes'] #['disease']#, 'gene/protein']
                        
            
            for etype in self.etypes_dd:
                src, dst = etype[0], etype[2]
                if src == 'disease':
                    all_disease_ids = torch.where(G.out_degrees(etype=etype) != 0)[0]
                elif dst == 'disease':
                    all_disease_ids = torch.where(G.in_degrees(etype=etype) != 0)[0]
                    
                if sim_measure == 'all_nodes_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, self.disease_etypes, self.disease_nodes) for i in all_disease_ids}
                elif sim_measure == 'all_nodes_profile_more':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, self.disease_etypes_all, self.disease_nodes_all) for i in all_disease_ids}
                elif sim_measure == 'protein_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein']) for i in all_disease_ids}
                elif sim_measure == 'protein_random_walk':
                    diseases_profile = {i.item(): obtain_protein_random_walk_profile(i, num_walks, path_length, G, self.disease_etypes, self.disease_nodes, walk_mode) for i in all_disease_ids}
                elif sim_measure == 'bert':
                    diseases_profile = {i.item(): torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]]) for i in all_disease_ids}
                elif sim_measure == 'profile+bert':
                    diseases_profile = {i.item(): torch.cat((obtain_disease_profile(G, i, self.disease_etypes, self.disease_nodes), torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]]))) for i in all_disease_ids}
                    
                diseaseid2id = dict(zip(all_disease_ids.detach().cpu().numpy(), range(len(all_disease_ids))))
                disease_profile_tensor = torch.stack([diseases_profile[i.item()] for i in all_disease_ids])
                sim_all = sim_matrix(disease_profile_tensor, disease_profile_tensor)
                
                self.sim_all_etypes[etype] = sim_all
                self.diseaseid2id_etypes[etype] = diseaseid2id
                self.diseases_profile_etypes[etype] = diseases_profile
                
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        rel_idx = self.rel2idx[edges._etype]
        h_r = self.W[rel_idx]
        score = torch.sum(h_u * h_r * h_v, dim=1)
        return {'score': score}

    def forward(self, graph, G, h, pretrain_mode, mode, block = None, only_relation = None):
        with graph.local_scope():
            scores = {}
            s_l = []
            
            if len(graph.canonical_etypes) == 1:
                etypes_train = graph.canonical_etypes
            else:
                etypes_train = self.etypes_dd
            
            if only_relation is not None:
                if only_relation == 'indication':
                    etypes_train = [('drug', 'indication', 'disease'),
                                    ('disease', 'rev_indication', 'drug')]
                elif only_relation == 'contraindication':
                    etypes_train = [('drug', 'contraindication', 'disease'), 
                                   ('disease', 'rev_contraindication', 'drug')]
                elif only_relation == 'off-label':
                    etypes_train = [('drug', 'off-label use', 'disease'),
                                   ('disease', 'rev_off-label use', 'drug')]
                else:
                    return ValueError
            
            graph.ndata['h'] = h
            
            if pretrain_mode:
                # during pretraining....
                etypes_all = [i for i in graph.canonical_etypes if graph.edges(etype = i)[0].shape[0] != 0]
                for etype in etypes_all:
                    graph.apply_edges(self.apply_edges, etype=etype)    
                    out = torch.sigmoid(graph.edges[etype].data['score'])
                    s_l.append(out)
                    scores[etype] = out
            else:
                # finetuning on drug disease only...
                
                for etype in etypes_train:

                    if self.proto:
                        src, dst = etype[0], etype[2]
                        src_rel_idx = torch.where(graph.out_degrees(etype=etype) != 0)
                        dst_rel_idx = torch.where(graph.in_degrees(etype=etype) != 0)
                        src_h = h[src][src_rel_idx]
                        dst_h = h[dst][dst_rel_idx]

                        src_rel_ids_keys = torch.where(G.out_degrees(etype=etype) != 0)
                        dst_rel_ids_keys = torch.where(G.in_degrees(etype=etype) != 0)
                        src_h_keys = h[src][src_rel_ids_keys]
                        dst_h_keys = h[dst][dst_rel_ids_keys]

                        h_disease = {}

                        if src == 'disease':
                            h_disease['disease_query'] = src_h
                            h_disease['disease_key'] = src_h_keys
                            h_disease['disease_query_id'] = src_rel_idx
                            h_disease['disease_key_id'] = src_rel_ids_keys
                        elif dst == 'disease':
                            h_disease['disease_query'] = dst_h
                            h_disease['disease_key'] = dst_h_keys
                            h_disease['disease_query_id'] = dst_rel_idx
                            h_disease['disease_key_id'] = dst_rel_ids_keys

                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'protein_random_walk', 'bert', 'profile+bert', 'all_nodes_profile_more']:

                            try:
                                sim = self.sim_all_etypes[etype][np.array([self.diseaseid2id_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]])]
                            except:
                                
                                #disease_etypes = ['disease_disease'] #, 'rev_disease_protein']
                                #disease_nodes = ['disease'] #, 'gene/protein']
                                #disease_etypes_all = ['disease_disease', 'disease_phenotype_positive', 'rev_exposure_disease'] #, 'rev_disease_protein' 
                                #disease_nodes_all = ['disease', 'effect/phenotype', 'exposure'] #'gene/protein',
                                ## new disease not seen in the training set
                                for i in h_disease['disease_query_id'][0]:
                                    if i.item() not in self.diseases_profile_etypes[etype]:
                                        if self.sim_measure == 'all_nodes_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, self.disease_etypes, self.disease_nodes)
                                        elif self.sim_measure == 'all_nodes_profile_more':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, self.disease_etypes_all, self.disease_nodes_all)    
                                        elif self.sim_measure == 'protein_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein'])
                                        elif self.sim_measure == 'protein_random_walk':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_protein_random_walk_profile(i, self.num_walks, self.path_length, G, self.disease_etypes, self.disease_nodes, self.walk_mode)
                                        elif self.sim_measure == 'bert':
                                            self.diseases_profile_etypes[etype][i.item()] = torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]])
                                        elif self.sim_measure == 'profile+bert':
                                            self.diseases_profile_etypes[etype][i.item()] = torch.cat((obtain_disease_profile(G, i, self.disease_etypes, self.disease_nodes), torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]])))
                                            
                                profile_query = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]]
                                profile_query = torch.cat(profile_query).view(len(profile_query), -1)

                                profile_keys = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_key_id'][0]]
                                profile_keys = torch.cat(profile_keys).view(len(profile_keys), -1)

                                sim = sim_matrix(profile_query, profile_keys)

                            if src_h.shape[0] == src_h_keys.shape[0]:
                                ## during training...
                                coef = torch.topk(sim, self.k + 1).values[:, 1:]
                                coef = F.normalize(coef, p=1, dim=1)
                                embed = h_disease['disease_key'][torch.topk(sim, self.k + 1).indices[:, 1:]]
                            else:
                                ## during evaluation...
                                coef = torch.topk(sim, self.k).values[:, :]
                                coef = F.normalize(coef, p=1, dim=1)
                                embed = h_disease['disease_key'][torch.topk(sim, self.k).indices[:, :]]
                            out = torch.mul(embed, coef.unsqueeze(dim = 2).to(self.device)).sum(dim = 1)

                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'all_nodes_profile_more', 'protein_random_walk', 'bert', 'profile+bert']:
                            # for protein profile, we are only looking at diseases for now...
                            if self.agg_measure == 'learn':
                                coef_all = self.m(self.W_gate['disease'](torch.cat((h_disease['disease_query'], out), dim = 1)))
                                proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                            elif self.agg_measure == 'heuristics-0.8':
                                proto_emb = 0.8*h_disease['disease_query'] + 0.2*out
                            elif self.agg_measure == 'avg':
                                proto_emb = 0.5*h_disease['disease_query'] + 0.5*out
                            elif self.agg_measure == 'rarity':
                                if src == 'disease':
                                    coef_all = exponential(G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                elif dst == 'disease':
                                    coef_all = exponential(G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                            elif self.agg_measure == '100proto':
                                proto_emb = out
                            h['disease'][h_disease['disease_query_id']] = proto_emb
                        else:
                            if self.agg_measure == 'learn':
                                coef_src = self.m(self.W_gate[src](torch.cat((src_h, sim_emb_src), dim = 1)))
                                coef_dst = self.m(self.W_gate[dst](torch.cat((dst_h, sim_emb_dst), dim = 1)))
                            elif self.agg_measure == 'rarity':
                                # give high weights to proto embeddings for nodes that have low degrees
                                coef_src = exponential(G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                coef_dst = exponential(G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                            elif self.agg_measure == 'heuristics-0.8':
                                coef_src = 0.2
                                coef_dst = 0.2
                            elif self.agg_measure == 'avg':
                                coef_src = 0.5
                                coef_dst = 0.5
                            elif self.agg_measure == '100proto':
                                coef_src = 1
                                coef_dst = 1

                            proto_emb_src = (1 - coef_src)*src_h + coef_src*sim_emb_src
                            proto_emb_dst = (1 - coef_dst)*dst_h + coef_dst*sim_emb_dst

                            h[src][src_rel_idx] = proto_emb_src
                            h[dst][dst_rel_idx] = proto_emb_dst

                        graph.ndata['h'] = h

                    graph.apply_edges(self.apply_edges, etype=etype)    
                    out = graph.edges[etype].data['score']
                    s_l.append(out)
                    scores[etype] = out

                    if self.proto:
                        # recover back to the original embeddings for other relations
                        h[src][src_rel_idx] = src_h
                        h[dst][dst_rel_idx] = dst_h
                
                
            if pretrain_mode:
                s_l = torch.cat(s_l)             
            else: 
                s_l = torch.cat(s_l).reshape(-1,).detach().cpu().numpy()
            return scores, s_l


    
class AttHeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(AttHeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        
        self.attn_fc = nn.ModuleDict({
                name : nn.Linear(out_size * 2, 1, bias = False) for name in etypes
            })
    
    def edge_attention(self, edges):
        src_type = edges._etype[0]
        etype = edges._etype[1]
        dst_type = edges._etype[2]
        try:
            if src_type == dst_type:
                #print(edges)
                wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]], dim=1)
            else:
                if etype[:3] == 'rev':
                    wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]], dim=1)
                else:
                    wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]], dim=1)
        except:
            print(edges.src.keys())
            print(edges.dst.keys())
            raise ValueError
        a = self.attn_fc[etype](wh2)
        return {'e_%s' % etype: F.leaky_relu(a)}

    def message_func(self, edges):
        etype = edges._etype[1]
        return {'m': edges.src['Wh_%s' % etype], 'e': edges.data['e_%s' % etype]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'h': h}
    
    def forward(self, G, feat_dict, return_att = False):
        with G.local_scope():        
            funcs = {}
            att = {}
            etypes_all = [i for i in G.canonical_etypes if G.edges(etype = i)[0].shape[0] != 0]
            for srctype, etype, dsttype in etypes_all:
                Wh = self.weight[etype](feat_dict[srctype])
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
            
            for srctype, etype, dsttype in etypes_all:
                try:
                    G.apply_edges(self.edge_attention, etype=etype)
                except:
                    print(etype)
                    # Assuming 'etype' is your edge type of interest
                    src, dst, eid = G.edges(etype=etype, form='all')

                    print(src)
                    print(dst)
                    print(f"Edge type: {etype}")
                    print(f"Source type: {srctype} Keys:", G.nodes[srctype].data.keys())
                    print(f"Destination type: {dsttype} Keys:", G.nodes[dsttype].data.keys())
                    if G.nodes[srctype].data:
                        print("Keys:", G.nodes[srctype].data.keys())
                    if G.nodes[dsttype].data:
                        print("Keys:", G.nodes[dsttype].data.keys())
                    raise ValueError
                if return_att:
                    att[(srctype, etype, dsttype)] = G.edges[etype].data['e_%s' % etype].detach().cpu().numpy()
                funcs[etype] = (self.message_func, self.reduce_func)
                
            G.multi_update_all(funcs, 'sum')
            
            return {ntype : G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())}, att
    
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        self.in_size = in_size
        self.out_size = out_size
            
        self.gate_storage = {}
        self.gate_score_storage = {}
        self.gate_penalty_storage = {}
            
    def add_graphmask_parameter(self, gate, baseline, layer):
        self.gate = gate
        self.baseline = baseline
        self.layer = layer
        
    def forward(self, G, feat_dict):
        funcs = {}
        etypes_all = [i for i in G.canonical_etypes if G.edges(etype = i)[0].shape[0] != 0]
        
        for srctype, etype, dsttype in etypes_all:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
       
        return {ntype : G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())}
 

    def gm_online(self, edges):
        etype = edges._etype[1]
        srctype = edges._etype[0]
        dsttype = edges._etype[2]
        
        if srctype == dsttype:
            gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]])
        else:
            if etype[:3] == 'rev':                
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]])
            else:
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]])
                
        #self.penalty += len(edges.src['Wh_%s' % etype])/self.num_of_edges * penalty
        #self.penalty += penalty
        self.penalty.append(penalty)
        
        self.num_masked += len(torch.where(gate.reshape(-1) != 1)[0])
        if self.no_base:
            message = gate.unsqueeze(-1) * edges.src['Wh_%s' % etype]
        else:
            message = gate.unsqueeze(-1) * edges.src['Wh_%s' % etype] + (1 - gate.unsqueeze(-1)) * self.baseline[etype][self.layer].unsqueeze(0)
        
        if self.return_gates:
            self.gate_storage[etype] = copy.deepcopy(gate.to('cpu').detach())
            self.gate_penalty_storage[etype] = copy.deepcopy(penalty_not_sum.to('cpu').detach())
            self.gate_score_storage[etype] = copy.deepcopy(gate_score.to('cpu').detach())
        return {'m': message}
    
    
    
    def message_func_no_replace(self, edges):
        etype = edges._etype[1]
        #self.msg_emb[etype] = edges.src['Wh_%s' % etype].to('cpu')
        return {'m': edges.src['Wh_%s' % etype]}
    
    
    def graphmask_forward(self, G, feat_dict, graphmask_mode, return_gates, no_base):
        self.no_base = no_base
        self.return_gates = return_gates
        self.penalty = []
        self.num_masked = 0
        self.num_of_edges = G.number_of_edges()
        
        funcs = {}
        etypes_all = G.canonical_etypes
        
        for srctype, etype, dsttype in etypes_all:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            
        for srctype, etype, dsttype in etypes_all:
            
            if graphmask_mode:
                ## replace the message!
                funcs[etype] = (self.gm_online, fn.mean('m', 'h'))
            else:
                ## normal propagation!
                funcs[etype] = (self.message_func_no_replace, fn.mean('m', 'h'))
                
        G.multi_update_all(funcs, 'sum')
        
        
        if graphmask_mode:
            self.penalty = torch.stack(self.penalty).reshape(-1,)
            #penalty_mean = torch.mean(self.penalty)
            #penalty_relation_reg = torch.sum(torch.log(self.penalty) * self.penalty)
            #penalty = penalty_mean + 0.1 * penalty_relation_reg
            penalty = torch.mean(self.penalty)
        else:
            penalty = 0 


        a = {}
        for ntype in G.ntypes:
            if 'h' in G.nodes[ntype].data:
                a[ntype] = G.nodes[ntype].data['h']
            else:
                print(f"Skipping node type {ntype} — no 'h' feature.")

        # a = {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
        return a, penalty, self.num_masked
    
class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, attention, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device, data_map):
        super(HeteroRGCN, self).__init__()

        if attention:
            self.layer1 = AttHeteroRGCNLayer(in_size, hidden_size, G.etypes)
            self.layer2 = AttHeteroRGCNLayer(hidden_size, out_size, G.etypes)
        else:
            self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
            self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)
        
        self.w_rels = nn.Parameter(torch.Tensor(len(G.canonical_etypes), out_size))
        nn.init.xavier_uniform_(self.w_rels, gain=nn.init.calculate_gain('relu'))
        rel2idx = dict(zip(G.canonical_etypes, list(range(len(G.canonical_etypes)))))
               
        self.pred = DistMultPredictor(n_hid = hidden_size, w_rels = self.w_rels, G = G, rel2idx = rel2idx, proto = proto, proto_num = proto_num, sim_measure = sim_measure, bert_measure = bert_measure, agg_measure = agg_measure, num_walks = num_walks, walk_mode = walk_mode, path_length = path_length, split = split, data_folder = data_folder, exp_lambda = exp_lambda, device = device, data_map=data_map)
        self.attention = attention
        
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.etypes = G.etypes
        self.device = device
        
    def forward_minibatch(self, pos_G, neg_G, blocks, G, mode = 'train', pretrain_mode = False):
        input_dict = blocks[0].srcdata['inp']
        h_dict = self.layer1(blocks[0], input_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h = self.layer2(blocks[1], h_dict)
        
        scores, out_pos = self.pred(pos_G, G, h, pretrain_mode, mode = mode + '_pos', block = blocks[1])
        scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg', block = blocks[1])
        return scores, scores_neg, out_pos, out_neg
        
    
    def forward(self, G, neg_G, eval_pos_G = None, return_h = False, return_att = False, mode = 'train', pretrain_mode = False):
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}

            if self.attention:
                h_dict, a_dict_l1 = self.layer1(G, input_dict, return_att)
                h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
                h, a_dict_l2 = self.layer2(G, h_dict, return_att)
            else:
                h_dict = self.layer1(G, input_dict)
                h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
                h = self.layer2(G, h_dict)

            if return_h:
                return h

            if return_att:
                return a_dict_l1, a_dict_l2

            # full batch
            if eval_pos_G is not None:
                # eval mode
                scores, out_pos = self.pred(eval_pos_G, G, h, pretrain_mode, mode = mode + '_pos')
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
            else:
                scores, out_pos = self.pred(G, G, h, pretrain_mode, mode = mode + '_pos')
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
    
    def graphmask_forward(self, G, pos_graph, neg_graph, graphmask_mode = False, return_gates = False, only_relation = None, no_base = False):
                    
        
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
            h_dict_l1, penalty_l1, num_masked_l1 = self.layer1.graphmask_forward(G, input_dict, graphmask_mode, return_gates, no_base)
            h_dict = {k : F.leaky_relu(h) for k, h in h_dict_l1.items()}
            h, penalty_l2, num_masked_l2 = self.layer2.graphmask_forward(G, h_dict, graphmask_mode, return_gates, no_base)         
            
            scores_pos, out_pos = self.pred(pos_graph, G, h, False, mode = 'train_pos', only_relation = only_relation)
            scores_neg, out_neg = self.pred(neg_graph, G, h, False, mode = 'train_neg', only_relation = only_relation)
            return scores_pos, scores_neg, penalty_l1 + penalty_l2, [num_masked_l1, num_masked_l2]

    
    def enable_layer(self, layer, graphmask = True):
        print("Enabling layer "+str(layer))
        
        for name in self.etypes:
            if graphmask:
                for parameter in self.gates_all[name][layer].parameters():
                    parameter.requires_grad = True
                self.baselines_all[name][layer].requires_grad = True
            else:
                for parameter in self.gates_all[name].parameters():
                    parameter.requires_grad = True
    
    def count_layers(self):
        return 2
    
    def get_gates(self):
        return [self.layer1.gate_storage, self.layer2.gate_storage]
    
    def get_gates_scores(self):
        return [self.layer1.gate_score_storage, self.layer2.gate_score_storage]
    
    def get_gates_penalties(self):
        return [self.layer1.gate_penalty_storage, self.layer2.gate_penalty_storage]
    
    
    def add_graphmask_parameters(self, G, threshold = 0.5, remove_key_parts = False, use_top_k = False, k = 0.05, gate_hidden_size = 32):
        gates_all, baselines_all = {}, {}
        hidden_size = self.hidden_size
        out_size = self.out_size
        print('gate_hidden_size: ', gate_hidden_size)
        for name in G.etypes:
            ## for each relation type

            gates = []
            baselines = []

            vertex_embedding_dims = [hidden_size, out_size]
            message_dims = [hidden_size, out_size]
            h_dims = message_dims

            for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
                gate_input_shape = [m_dim, m_dim]
                
                ### different layers have different gates
                gate = torch.nn.Sequential(
                    MultipleInputsLayernormLinear(gate_input_shape, gate_hidden_size),
                    nn.ReLU(),
                    nn.Linear(gate_hidden_size, 1),
                    Squeezer(),
                    SoftConcrete(threshold, remove_key_parts, use_top_k, k)
                )

                gates.append(gate)

                baseline = torch.FloatTensor(m_dim)
                stdv = 1. / math.sqrt(m_dim)
                baseline.uniform_(-stdv, stdv)
                baseline = torch.nn.Parameter(baseline, requires_grad=True)

                baselines.append(baseline)

            gates = torch.nn.ModuleList(gates)
            gates_all[name] = gates

            baselines = torch.nn.ParameterList(baselines)
            baselines_all[name] = baselines

        self.gates_all = nn.ModuleDict(gates_all)
        self.baselines_all = nn.ModuleDict(baselines_all)

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False
            
        self.layer1.add_graphmask_parameter(self.gates_all, self.baselines_all, 0)
        self.layer2.add_graphmask_parameter(self.gates_all, self.baselines_all, 1)