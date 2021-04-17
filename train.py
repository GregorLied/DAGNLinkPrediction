import time
import collections
import numpy as np
from model import DAGNLinkPrediction
import torch
from torch.optim.lr_scheduler import ExponentialLR

def train(args, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Use device: ", device)

    # Load data
    n_entities, n_relations = data[0], data[1]
    all_data, train_data, eval_data, test_data = data[2], data[3], data[4], data[5]
    edge_index, edge_type = data[6].to(device), data[7].to(device)
    
    # Preprocess data for training
    train_dict = get_data_dict(train_data)
    # Preprocess data for evaluation
    all_data_dict = get_data_dict(all_data)

    # Initialize model
    model = DAGNLinkPrediction(args, n_entities, n_relations, edge_index, edge_type).to(device)
    params = [value.numel() for value in model.parameters()]
    print("Number of paramers:", np.sum(params))

    # Set Adam optimizer and Cross Entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.l2_weight)
    if args.decay_rate:
        scheduler = ExponentialLR(optimizer, args.decay_rate)

    if args.loss == 'BCE':
        print('Use BCE as criterion')
        criterion = torch.nn.BCELoss()
    else:
        print('Use KLDivLoss as criterion')
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    print("Start training...")
    best_mrr, best_epoch = 0, 0
    for epoch in range(args.n_epochs):

        # Training
        t0 = time.time()
        np.random.shuffle(train_data)

        # skip the last incomplete minibatch if its size < batch size
        start, train_loss = 0, 0
        while start + args.batch_size <= len(train_data):
            model.train()
            optimizer.zero_grad()
            
            # Get batch
            data_batch, targets = get_train_batch(train_data, train_dict, n_entities, start, start + args.batch_size)
            data_batch, targets = data_batch.to(device), targets.to(device)

            # Forward Pass
            entity_embed, relation_embed = model.encode()
            preds = model.decode(args, entity_embed, relation_embed, data_batch)

            # Label Smoothing
            if args.label_smoothing:
                targets = ((1.0-args.label_smoothing)*targets) + (1.0/targets.size(1))

            # Backward Pass
            if args.loss == 'BCE':
                loss = criterion(preds, targets)
            else:
                # preds must be log-probabilities, targets must be probabilities
                # See also https://discuss.pytorch.org/t/kullback-leibler-divergence-loss-function-giving-negative-values/763/8
                loss = criterion(preds, F.softmax(targets, dim=1))
            loss.backward()

            # Update parameters
            optimizer.step()
            if args.decay_rate:
                scheduler.step()

            train_loss += loss.item()
            start += args.batch_size

        train_time = time.time() - t0

        # Evaluation
        t1 = time.time()
        eval_mr, eval_mrr, eval_hits = evaluate(args, model, eval_data, all_data_dict)
        eval_time = time.time() - t1
        t2 = time.time()
        test_mr, test_mrr, test_hits = evaluate(args, model, test_data, all_data_dict)
        test_time = time.time() - t2

        # Log
        log_eval = f"MR: {eval_mr:<8.3f} MRR: {eval_mrr:.3f} Hits@1: {eval_hits[0]:.3f} Hits@3: {eval_hits[2]:.3f} Hits@10: {eval_hits[9]:.3f}"
        log_test = f"MR: {test_mr:<8.3f} MRR: {test_mrr:.3f} Hits@1: {test_hits[0]:.3f} Hits@3: {test_hits[2]:.3f} Hits@10: {test_hits[9]:.3f}"
        log = f"epoch {epoch:<3} [{train_time:.1f}s + {eval_time:.1f}s + {test_time:.1f}s] | loss: {train_loss:<8.4f} | {log_eval} | {log_test} |"
        print(log)
        with open(args.log, 'a') as f:
            f.write(log + '\n')
        
        # Early stopping
        best_mrr, best_epoch, should_stop = early_stopping(eval_mrr, epoch, best_mrr, best_epoch)
        if should_stop:
            early_stopping_log = "Early Stopping triggered. Best epoch: epoch %d." % best_epoch
            print(early_stopping_log)
            with open(args.log, 'a') as f:
                 f.write(early_stopping_log)
            break

def get_train_batch(data, data_dict, n_entities, start, end):
    # Get batch of (head, relation, tail)-triples
    data_batch = data[start:end]
    # Get targets, where each row corresponds to i-th (head, relation)-pair
    # Set j-th column to 1, if j-th entity is a valid tail for the (head, relation)-pair
    # Set j-th column to 0, otherwise
    targets = np.zeros((len(data_batch), n_entities))
    for idx, triple in enumerate(data_batch):
        pair = triple[:2]  
        targets[idx, data_dict[pair]] = 1.
    # Create tensor
    data_batch = torch.tensor(data_batch)
    targets = torch.FloatTensor(targets)
    return data_batch, targets

def get_eval_batch(data, start, end):
    # Get batch of (head, relation, tail)-triples
    data_batch = data[start:end]
    # Create tensor
    data_batch = torch.tensor(data_batch)
    return data_batch

@torch.no_grad()
def evaluate(args, model, data, all_data_dict):
    model.eval()
    hits = []
    ranks = []
    for i in range(10):
        hits.append([])

    # Get trained embeddings
    entity_embed, relation_embed = model.encode()

    start = 0
    while start + args.batch_size <= len(data):
        # Get prediction
        data_batch = get_eval_batch(data, start, start + args.batch_size)
        data_batch = data_batch.to(model.device)
        preds = model.decode(args, entity_embed, relation_embed, data_batch)

        # Clean prediction results
        # Note: We stick to the same formulation as in ConvE, however I think that some prediction results get surpressed,
        # by the filters, if we got at least two triples with the same (head, relation)-pair in the same batch
        # E.g (Emma Watson, actor_in, Harry Potter), (Daniel Radcliff, actor_in, Harry Potter)
        # UPDATE: This is not a problem, as the prediction for (Emma Watson, actor_in, Harry Potter) will be 
        # in another line j then (Daniel Radcliff, actor_in, Harry Potter) in preds
        for j in range(data_batch.shape[0]):
            # get head, relation, tail from each triple in batch
            head = data_batch[j][0]
            relation = data_batch[j][1]
            tail = data_batch[j][2]
            # filter contains ALL tails from train, valid, test set respectively for a given (head, relation)-pair
            filter = all_data_dict[(head, relation)]
            # save the prediction that is relevant
            target_value = preds[j, tail].item()
            # zero all known cases from train, valid, test set
            preds[j, filter] = 0.0
            # only consider saved predictions of the current set
            preds[j, tail] = target_value

        # Evaluate prediction results
        sort_values, sort_idxs = torch.sort(preds, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()
        for j in range(data_batch.shape[0]):
            tail = data_batch[j][2]
            rank = np.where(sort_idxs[j]==tail.item())[0][0]
            ranks.append(rank+1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        start += args.batch_size
        
    mr = np.mean(ranks)
    mrr = np.mean(1./np.array(ranks))
    hits = np.mean(hits, axis=1)

    return mr, mrr, hits

def early_stopping(cur_mrr, cur_epoch, best_mrr, best_epoch, stopping=50):
    if cur_mrr > best_mrr:
        best_mrr = cur_mrr
        best_epoch = cur_epoch
    if cur_epoch - best_epoch >= stopping:
        print("Early Stopping triggered. No improvements since %d epochs." % stopping)
        should_stop = True
    else:
        should_stop = False
    return best_mrr, best_epoch, should_stop

def get_data_dict(data):
    data_dict = collections.defaultdict(list)
    for triple in data:
        data_dict[(triple[0], triple[1])].append(triple[2])
    return data_dict