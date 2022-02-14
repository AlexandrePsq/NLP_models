import os
import sys
import time
import math
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import utils
from data import Corpus
from modeling_hacked_lstm import RNNModel
from lstm_utils import get_batch, repackage_hidden, batchify, save, load, get_preference_params, read_yaml, save_yaml

import matplotlib.pyplot as plt
plt.switch_backend('agg')

params = get_preference_params()
path2derivatives = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives'


###############################################################################
# Evaluating code
###############################################################################

def evaluate(model, criterion, ntokens, data_source, eval_batch_size):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in tqdm(range(0, data_source.size(0) - 1, params['bptt'])):
            data, targets = get_batch(data_source, i, params['bptt'])
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)



###############################################################################
# Training code
###############################################################################

def forward(model, train_data, corpus, criterion, epoch, lr, bsz=params['bsz'], data_name='wikipedia', language='english', path2derivatives='./', extra_name=''):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(params['bsz'])
    for batch, i in tqdm(enumerate(range(0, train_data.size(0) - 1, params['bptt']))):
        data, targets = get_batch(train_data, i, params['bptt'])
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-lr)

        total_loss += loss.item()
        
        if ((epoch in [1, 2, 3] and batch in [1, 2, 5, 10, 15, 30, 50, 100, 200, 500, 1000, 5000]) or batch%10000==0) and epoch < 15:
            save(model, data_name, language, path2derivatives, extra_name=extra_name+f'_checkpoint_epoch-{epoch}_batch-{batch}')

        if batch % params['log_interval'] == 0 and batch > 0:
            cur_loss = total_loss / params['log_interval']
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // params['bptt'], lr,
                elapsed * 1000 / params['log_interval'], cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def train(config_path, data, data_name, language, eval_batch_size=params['eval_batch_size'], bsz=params['bsz'], epochs=params['epochs'], start_from_scratch=False, extra_name='', train_model=True):
    torch.manual_seed(params['seed']) # setting seed for reproductibility
    device = torch.device("cuda" if params['cuda'] else "cpu")
    corpus = Corpus(data, language)
    if train_model:
        train_data = batchify(corpus.train, bsz, device)
        val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, bsz, device)

    # Build the model
    ntokens = len(corpus.dictionary)
    model_parameters = read_yaml(config_path)
    model_parameters['ntoken'] = ntokens
    save_yaml(model_parameters, config_path)
    model = RNNModel(**model_parameters)
    #model.encoder.num_embeddings = ntokens
    #model.decoder.out_features = ntokens
    if train_model:
        model.vocab = corpus.dictionary
    try:
        print('loading best saved model...')
        if start_from_scratch:
            print("Be sure you deleted any saved version of the model...")
            raise ValueError
        else:
            model = load(model, data_name, language, path2derivatives, extra_name=extra_name, parameters=model_parameters, device=device)
    except:
        print("Couldn't load pre-trained model...")
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    best_val_loss = None
    best_epoch = None
    lr = params['lr']
    valid_ppl = []

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        if not train_model:
            print("Only evaluating the model. No training is done.")
            raise KeyboardInterrupt
        print('Entering training...')
        save(model, data_name, language, path2derivatives, extra_name=extra_name+f'_checkpoint_epoch-1_batch-0')
        for epoch in tqdm(range(1, epochs+4)):
            if epoch == epochs + 1: # we added 3 more epoch with a smaller lr for tuning
                lr /= 2.0
            if (epoch % 5 ==0) and epoch > 0:
                lr /= 2.0
            epoch_start_time = time.time()
            forward(model, train_data, corpus, criterion, epoch, lr=lr, data_name=data_name, language=language, path2derivatives=path2derivatives, extra_name=extra_name)
            val_loss = evaluate(model, criterion, ntokens, val_data, eval_batch_size)
            valid_ppl.append(math.exp(val_loss))
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                save(model, data_name, language, path2derivatives, extra_name=extra_name)
                best_val_loss = val_loss
                best_epoch = epoch
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    print('loading best saved model...')
    model = load(model, data_name, language, path2derivatives, extra_name=extra_name, parameters=model_parameters, device=device)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

    # Run on test data.
    print('evaluation...')
    test_loss = evaluate(model, criterion, ntokens, test_data, eval_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    path = '_'.join([model.__name__(), data_name, language]) + f'{extra_name}_test_loss.txt'
    path = os.path.join(path2derivatives, 'fMRI/models/', language, 'LSTM', path)
    utils.write(path, '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))

    if train_model:
        # plot the perplexity as a function of the number of epochs
        print(valid_ppl)
        ppl_df = pd.DataFrame(valid_ppl, columns=['ppl'])
        ppl_df.to_csv(os.path.join(path2derivatives, 'fMRI', 'models', language, '{}_{}_perplexity.csv'.format(model.__name__(), extra_name)))
        plt.plot(valid_ppl, color='b', linestyle='-')
        plt.ylim(0,100)
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='best epoch: {}'.format(best_epoch))
        plt.axhline(math.exp(best_val_loss), color='r', linestyle=':', label='best validation ppl: {}'.format(math.exp(best_val_loss)))
        plt.axhline(math.exp(test_loss), color='g', linestyle='-.', label='test ppl: {}'.format(math.exp(test_loss)))
        plt.xlabel('epoch number')
        plt.ylabel('validation perplexity')
        plt.legend()
        plt.savefig(os.path.join(path2derivatives, 'fMRI', 'models', language, '{}_{}_{}_perplexity.png'.format(model.__name__(), data_name, extra_name)))
    
    
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train LSTM model.')
    parser.add_argument("--language", type=str, default='english')
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--data_folder_path", type=str)
    parser.add_argument("--start_from_scratch", action='store_true')
    parser.add_argument("--extra_name", type=str, default='')
    parser.add_argument("--no_training", action='store_false')

    args = parser.parse_args()
    start_from_scratch = args.start_from_scratch if args.start_from_scratch is not None else False
    
    train(args.config_path, data=args.data_folder_path, data_name=args.data_name, language=args.language, start_from_scratch=start_from_scratch, extra_name=args.extra_name, train_model=args.no_training)
    print('--> Done')
