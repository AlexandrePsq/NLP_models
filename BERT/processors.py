import os
import time
import torch
import numpy as np
from utils import format_time

from dataset import Dataset, InputExample, InputFeatures
from metrics import Metrics
from utils import save



#########################################
############## Base Class ###############
#########################################

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, dataset_object):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, dataset_object):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, dataset_object):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self, dataset_object):
        """Gets the list of labels for this data set."""
        return dataset_object.get_labels()
    
    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        raise NotImplementedError()

    def get_data_loader(self, features, batch_size, local_rank):
        """Return data loader object."""
        raise NotImplementedError()


class ModelProcessor(object):
    """Base class for model training/validation and evaluation."""

    def __init__(self, 
                    model, 
                    optimizer=None, 
                    tokenizer=None, 
                    scheduler=None, 
                    device=None, 
                    metric_name=None, 
                    nb_epochs=3, 
                    use_output_mask=False, 
                    nb_checkpoints=24):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device
        self.nb_checkpoints = nb_checkpoints
        self.nb_epochs = nb_epochs
        self.metric_name = metric_name
        self.use_output_mask = use_output_mask


    #########################################
    ########### Training functions ##########
    #########################################

    def training_step(self, batch, total_train_loss):
        """ Compute a training step in a model training.
        Arguments:
            - batch:
            - total_train_loss:
        Returns:
            - total_train_loss: (float) accumulated loss from the batch 
            and the input
        """
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        input_ids = batch[0].to(self.device)
        attention_mask = batch[1].to(self.device)
        token_type_ids = batch[2].to(self.device)
        labels_ids = batch[3].to(self.device)

        self.model.zero_grad()        

        # The documentation for the BERT `model` are here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html
        outputs = self.model(input_ids, 
                            token_type_ids=token_type_ids, 
                            attention_mask=attention_mask, 
                            labels=labels_ids)
        loss = outputs[0] 

        # The `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        self.optimizer.step()
        # Update the learning rate.
        self.scheduler.step()
        return total_train_loss

    def train(self, train_dataloader, validation_dataloader, output_dir):
        """ Train a model with evaluation at each step, given an optimizer, scheduler, device and train and 
        validation data loaders.
        Returns loss statistics from training and evaluations.
        """
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        save_step = self.nb_epochs * len(train_dataloader) // self.nb_checkpoints

        # For each epoch...
        for epoch_i in range(0, self.nb_epochs):
            print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.nb_epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            # index of the checkpoints
            checkpoints_index = 1 + epoch_i * (len(train_dataloader) // save_step)
            # Reset the total loss for this epoch.
            total_train_loss = 0
            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                # Save model weights to have a given number of checkpoints at the end
                if step != 0 and step % save_step == 0:
                    save(self.model, self.tokenizer, output_dir, 'checkpoint_' + str(checkpoints_index))
                    checkpoints_index += 1
                # Progress update every 40 batches.
                if step % 50 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                total_train_loss = self.training_step(batch, total_train_loss)

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
                
            avg_val_accuracy, avg_val_loss, validation_time, report = self.evaluate(validation_dataloader)        

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time,
                    'report': report
                }
            )
        print("\nTraining complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        return training_stats

    def evaluate(self, dataloader):
        """ Evaluate a model on a validation dataloader.
        """
        print("Running Validation...")
        t0 = time.time()
        self.model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        y_true = []
        y_pred = []

        # Evaluate data for one epoch
        for batch in dataloader:
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: token_type_ids
            #   [3]: labels 
            #   [4]: output_mask (optional)
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            token_type_ids = batch[2].to(self.device)
            label_ids = batch[3].to(self.device)
            output_mask = None 
            if self.use_output_mask:
                output_mask = batch[4].to(self.device)
            
            with torch.no_grad():        
                # The documentation for the BERT `models` are here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html
                outputs = self.model(input_ids, 
                                    token_type_ids=token_type_ids, 
                                    attention_mask=attention_mask,
                                    labels=label_ids)
            loss = outputs[0] 
            logits = outputs[1]
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            if self.use_output_mask:
                output_mask = output_mask.to('cpu').numpy()
                active_loss = (output_mask == 1)
            else:
                active_loss = np.ones(label_ids.shape)
                active_loss = (active_loss == 1)
            pred_flat = np.argmax(logits, axis=-1)[active_loss].flatten()
            labels_flat = label_ids[active_loss].flatten()
            y_true.append(labels_flat)
            y_pred.append(pred_flat)

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += Metrics.flat_accuracy(label_ids, logits)

        # Report results
        report = Metrics.report(self.metric_name, 
                                [item for sublist in y_true for item in sublist], 
                                [item for sublist in y_pred for item in sublist])
        print(report)
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        return avg_val_accuracy, avg_val_loss, validation_time, report
            