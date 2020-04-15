""" This training code is based on the `run_glue.py` script here:
https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
"""


import os
import wget
import time
import yaml
import glob
import torch
import random
import inspect
import datetime
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForQuestionAnswering, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import BertForNextSentencePrediction, BertForSequenceClassification, BertForTokenClassification
from transformers import BertTokenizer, BertModel, BertForPreTraining, BertForMaskedLM, WEIGHTS_NAME, CONFIG_NAME

from utils import read_yaml, set_seed, format_time, filter_args, get_device, fetch_dataset_from_url, fetch_data, save


#########################################
### Inputs/Features related functions ###
#########################################

def get_inputs_tensors(sentences, labels, tokenizer, max_length=128):
    """ Tokenize all of the sentences and map the tokens 
    to thier word IDs.
    `encode_plus` will:
        (1) Tokenize the sentence.
        (2) Prepend the `[CLS]` token to the start.
        (3) Append the `[SEP]` token to the end.
        (4) Map tokens to their IDs.
        (5) Pad or truncate the sentence to `max_length`
        (6) Create attention masks for [PAD] tokens.
    Returns the lists of tensors (inputs_ids, attention_masks, 
    labels).
    """
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def get_data_loaders(input_ids, attention_masks, labels, train_size_percentage=0.9, batch_size=32):
    """ Returns Pytorch train and validation data loader.
    """
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)
    # Compute the number of samples to include in each set.
    train_size = int(train_size_percentage * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # The DataLoader needs to know our batch size for training
    batch_size = batch_size
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size 
            )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size 
            )
            
    return train_dataloader, validation_dataloader

def extract_inputs(data, corpus_name):
    """ Extract non-tokenized sentences and labels from data.
    """
    if corpus_name in ['COLA']:
        sentences = data.sentence.values
        labels = data.label.values
    return sentences, labels


#########################################
########### Measures functions ##########
#########################################

def flat_accuracy(preds, labels):
    """ Function to calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#########################################
########### Training functions ##########
#########################################

def training_step(model, optimizer, scheduler, batch, device, total_train_loss):
    """ Compute a training step in a model training.
    Arguments:
        - model:
        - optimizer:
        - scheduler:
        - batch:
        - device:
        - total_train_loss:
    Returns:
        - total_train_loss: (float) accumulated loss from the batch 
        and the input
    """
    # `batch` contains three pytorch tensors:
    #   [0]: input ids 
    #   [1]: attention masks
    #   [2]: labels 
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    model.zero_grad()        

    # The documentation for the BERT `model` are here: 
    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html
    loss, logits = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

    # The `.item()` function just returns the Python value 
    # from the tensor.
    total_train_loss += loss.item()
    # Perform a backward pass to calculate the gradients.
    loss.backward()
    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Update parameters and take a step using the computed gradient.
    # The optimizer dictates the "update rule"--how the parameters are
    # modified based on their gradients, the learning rate, etc.
    optimizer.step()
    # Update the learning rate.
    scheduler.step()
    return total_train_loss

def train(model, train_dataloader, validation_dataloader, optimizer, scheduler, device, nb_epochs=3):
    """ Train a model with evaluation at each step, given an optimizer, scheduler, device and train and 
    validation data loaders.
    Returns loss statistics from training and evaluations.
    """
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, nb_epochs):
        print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, nb_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            avg_train_loss = training_step(model, optimizer, scheduler, batch, device, total_train_loss)

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        avg_val_accuracy, avg_val_loss, validation_time = evaluate(model, validation_dataloader, device)        

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return training_stats

def evaluate(model, validation_dataloader, device):
    """ Evaluate a model on a validation dataloader.
    """
    print("Running Validation...")
    t0 = time.time()
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        
            # The documentation for the BERT `models` are here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html
            (loss, logits) = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach()
        label_ids = b_labels
        if device=='cuda':
            logits = logits.cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    return avg_val_accuracy, avg_val_loss, validation_time
        

#########################################
########## Reporting functions ##########
#########################################

def plots_train_val_loss(training_stats, nb_epochs):
    """ Plot train and validation losses over fine-tuning.
    """
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(training_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(training_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(np.arange(1, nb_epochs + 1))

    plt.show()




########################################################################################################
# ------------------------------------------- FINE - TUNING -------------------------------------------#
########################################################################################################

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Fine-tune a BERT model for a specific NLP task.")
    parser.add_argument('--yaml_file', type=str, help='''Path to the yaml file containing additional information on how 
                                                        the dataset is structured.''')
    args = parser.parse_args()

    # Fetch parameters
    parameters = read_yaml(args.yaml_file)
    # Set seed for reproductibility 
    set_seed(parameters['seed'])
    # Retrieve the device on which to run
    device = get_device()
    # Fetch data (training + validation) and parameters
    kwargs = filter_args(fetch_data, parameters)
    data = fetch_data(parameters['dataset'], **kwargs)
    # Fetch pre-trained Bert model and Tokenizer
    if parameters['task'] in ['POS-tagging', 'NER']:
        model = BertForTokenClassification.from_pretrained(
                    parameters['pretrained_model'],
                    num_labels=parameters['num_labels'], # The number of output labels for classification.  
                    output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                    output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
        )
    elif parameters['task'] in ['sentiment-analysis']:
        model = BertForSequenceClassification.from_pretrained(
                    parameters['pretrained_model'],
                    num_labels=parameters['num_labels'], # The number of output labels for classification.  
                    output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                    output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
        )
    tokenizer = BertTokenizer.from_pretrained(parameters['pretrained_tokenizer'])
    # Extract inputs from data
    sentences, labels = extract_inputs(data, parameters['corpus_name'])
    # Transforms former inputs to input tensors for Pytorch model
    input_ids, attention_masks, labels = get_inputs_tensors(sentences, labels, tokenizer, max_length=128)
    # Create data loaders
    train_dataloader, validation_dataloader = get_data_loaders(
                                                input_ids, 
                                                attention_masks, 
                                                labels, 
                                                train_size_percentage=parameters['train_size_percentage'], 
                                                batch_size=parameters['batch_size'])
    # Create optimizer and learning rate scheduler
    optimizer = AdamW(
                    model.parameters(),
                    lr=parameters['learning_rate'],
                    eps=parameters['adam_epsilon']
                )
    
    total_steps = len(train_dataloader) * parameters['nb_epochs'] # Total number of training steps is [nb batches] x [nb epochs]. 
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=parameters['num_warmup_steps'],
                    num_training_steps=total_steps
                )
    # Fine-tune the model
    training_stats = train(model,
                            train_dataloader, 
                            validation_dataloader, 
                            optimizer, 
                            scheduler, 
                            device, 
                            parameters['nb_epochs'])
    # Save fine-tuned model and save it
    save(model, tokenizer, parameters['output_dir'])
    plots_train_val_loss(training_stats, parameters['nb_epochs'])