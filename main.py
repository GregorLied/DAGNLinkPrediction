import os
import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)
        
def get_filepath(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return file_path

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='FB15k-237', help='Choose dataset from {FB15k-237, WN18RR}')
parser.add_argument('--reverse', type=bool, default=False, help='Use (t, r^-1, h)-triples aswell?')
parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
parser.add_argument('--dim', type=int, default=100, help='dimension of entity and relation embeddings')
parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension in feed-forward layer')
parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--pow_iter', type=int, default=4, help='number of power iterations')
parser.add_argument('--alpha', type=float, default= 0.2, help='PPR teleport probability')
parser.add_argument('--att_dropout', type=float, default= 0.1, help='attention dropout ratio')
parser.add_argument('--feat_dropout', type=float, default= 0.1, help='feature dropout ratio')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--loss', type=str, default='BCE', help='loss type')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-8, help='l2-reg')
parser.add_argument('--decay_rate', type=float, default=0.0, help='decay rate for LRScheduler')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')

args = parser.parse_args()

# Set seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# Ensure that all operations are deterministic on GPU for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Logging
save_dir = 'log/{}/layer{}_dim{}_l2{}_epochs{}_batch{}_lr{}/'.format(args.dataset, args.n_layers, args.dim, args.l2_weight , args.n_epochs, args.batch_size, args.lr)
ensureDir(save_dir)
args.log = get_filepath(save_dir)

# Traing / Eval / Test
data = load_data(args)
train(args, data)