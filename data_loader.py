import os
import argparse
import numpy as np
import torch
from time import time

import os
import numpy as np
import collections

def load_data(args):
    
    print("loading data...")
    
    all_data, train_data, eval_data, test_data, n_entities, n_relations, n_triples = load_dataset(args)
    edge_index, edge_type = get_adj(train_data)
    
    print(f"size of train, eval and test set: [{len(train_data), len(eval_data), len(test_data)}]")
    print(f"number of entities, relations and triples: [{n_entities, n_relations, n_triples}]")
    print("done.\n")

    return n_entities, n_relations, all_data, train_data, eval_data, test_data, edge_index, edge_type

def load_dataset(args):
    print('reading rating file ...')

    # reading rating file
    train_data = load_file(args.dataset, "train", args.reverse)
    eval_data = load_file(args.dataset, "valid", args.reverse)
    test_data = load_file(args.dataset, "test", args.reverse)
    all_data = train_data + eval_data + test_data

    # get list of all entities / relations
    entities = sorted(list(set([d[0] for d in all_data]+[d[2] for d in all_data])))
    relations = sorted(list(set([d[1] for d in all_data])))

    # remap entities / relations to unique identifier
    ent2id = {entities[i]:i for i in range(len(entities))}
    rel2id = {relations[i]:i for i in range(len(relations))}

    all_data = [(ent2id[all_data[i][0]], rel2id[all_data[i][1]], \
                 ent2id[all_data[i][2]]) for i in range(len(all_data))]
    train_data = [(ent2id[train_data[i][0]], rel2id[train_data[i][1]], \
                  ent2id[train_data[i][2]]) for i in range(len(train_data))]
    eval_data = [(ent2id[eval_data[i][0]], rel2id[eval_data[i][1]], \
                 ent2id[eval_data[i][2]]) for i in range(len(eval_data))]
    test_data = [(ent2id[test_data[i][0]], rel2id[test_data[i][1]], \
                 ent2id[test_data[i][2]]) for i in range(len(test_data))]

    # get data statistics
    n_entities = len(set([triple[0] for triple in all_data]) | set([triple[2] for triple in all_data]))
    n_relations = len(set([triple[1] for triple in all_data]))
    n_triples = len(all_data)

    return all_data, train_data, eval_data, test_data, n_entities, n_relations, n_triples

def load_file(dataset, split, reverse):
    with open("./data/{}/{}.txt".format(dataset, split), "r") as f:
        data = f.read().strip().split("\n")
        data = [i.split() for i in data]
        if reverse:
            data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
    return data

def get_adj(data):
    # Get Edge Index und Edge Type – Sorted makes sure that everything is coalesced
    edge_list = sorted(data, key=lambda x: (x[0], x[2], x[1]))
    edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index, edge_type = torch.cat([edge[0],edge[2]], dim=1), edge[1] 

    return edge_index, edge_type
