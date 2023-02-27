import os
import time
import math
import torch
import pickle
import pickle5 as pickle5
import scipy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler

from transformers import AdamW, get_linear_schedule_with_warmup

losses = {
    'lexical': 'POS --> nn.CrossEntropyLoss(), vocabulary_index --> nn.CrossEntropyLoss()',
    'next_word_prediction': 'input_ids[1:] --> nn.CrossEntropyLoss()',
    'X_future_words_prediction': 'np.vstack([input_ids[1:1-X], ..., input_ids[X:]]).T --> nn.CrossEntropyLoss() , X > 1',
    'X_past_words_prediction': 'np.vstack([input_ids[1:1-X], ..., input_ids[X:]]).T --> nn.CrossEntropyLoss()',
    'sentence_level_ordering': 'stack average hidden-state of X previous sentences, compute average hidden-state for current sentence and compute distance with other averaged hidden-sates, compute argsort of this distance and apply L1Loss with [X, X-1, ..., 2, 1]--> nn.L1Loss()',
}

config = {
    'hidden_size': 768,
    'loss': [nn.CrossEntropyLoss(), None],
    'vocab_size': 50257,
    'pad_token_id': 50256,
    'num_hidden_layers': 1,
    'layer_norm_epsilon': 1e-05,
    'num_attention_heads': 12,
    'attention_probs_dropout_prob': 0.1,
    'attention_res_dropout_prob': 0.1,
    'dropout': 0.1,
    'device': 'cpu',
    'dtype': torch.float64,
    'activation_function_i': 'sigmoid',
    'activation_function_c': 'sigmoid',
    'activation_function_f': 'sigmoid',
    'activation_function_o': 'sigmoid',
    'proj_head': [1, None],
    'log_interval': 200,
    'bsz': 100,
    'clip': 0.25, #1.0,
    'layer_norm_eps':  1e-05,
    'embedding_dropout_prob': 0.1,
    'learning_rate': 1e-3, #20
    'adam_epsilon': 1e-8,
    'num_warmup_steps': 0,
}
torch.autograd.set_detect_anomaly(True)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.embedding_dropout_prob)

    def forward(
        self, input_ids=None,
    ):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class AttentiveLSTM(nn.Module):
    """Class implementing the Attentive LSTM, a variant of transformer where the operation after the attention operation
    are replaced by LSTM cells.
    """
    def __init__(self, config):
        super().__init__()
        config = AttrDict(config)
        self.config = config
        self.current_sentence_index = None
        self.previous_sentence_hidden_state = []
        # Modules implementation
        self.embedding = Embeddings(config)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.loss_fcts = [config.loss[layer_index] for layer_index in range(config.num_hidden_layers)]
        self.losses = [0 for layer_index in range(config.num_hidden_layers)]
        self.proj_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size, bias=False).double() if config.proj_head[_] is not None else None for _ in range(config.num_hidden_layers)])
    
    def init_weights(self):
        """ Initialize the weights of the model using 
        an uniform distribution and zero for the bias 
        of the decoder.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, dataloader, network_hidden_states, network_cell_states):
        """
        Args:
            - input_id: str
            - network_hidden_states: list
            - network_cell_states: list
        Returns:
            - 
        """
        # attention à bien gérer les hidden-states dans le forward !
        start_time = time.time()
        #self.train()
        for index_batch, (batch, ground_truth) in enumerate(dataloader):
            input_hidden_states = self.embedding(batch)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            input_hidden_states = repackage_hidden(input_hidden_states)
            network_hidden_states = [repackage_hidden(item) for item in network_hidden_states]
            network_cell_states = [repackage_hidden(item) for item in network_cell_states]
            self.zero_grad()
            # attention à bien gérer les hidden-states dans le forward !
            for index_layer, (layer, loss, proj_head) in enumerate(zip(self.layers, self.loss_fcts, self.proj_heads)):
                hidden_states = [input_hidden_states, ] + network_hidden_states
                current_cell_state = network_cell_states[index_layer]
                current_hidden_state = network_hidden_states[index_layer]
                new_hidden, new_cell = layer(current_hidden_state, hidden_states, current_cell_state)
                network_hidden_states[index_layer] = new_hidden
                network_cell_states[index_layer] = new_cell
                if proj_head is not None:
                    predictions = proj_head(new_hidden)
                    loss_ = loss(predictions.view(-1, predictions.size(-1)), ground_truth.view(-1))
                    print(loss_)
                    #print(np.argmax(nn.Softmax(dim=-1)(predictions).detach().numpy(), axis=-1), ground_truth)
                    #if index_batch % 50 == 0 and index_batch > 0:
                    loss_.backward(retain_graph=True)
                    self.losses[index_layer] += loss_.item()

            nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)
            #loss_.backward()
            #for p in self.parameters():
            #    print(p.grad)
            #if index_batch % 50 == 0 and index_batch > 0:
            optimizer.step()
            scheduler.step()
            
            if index_batch % self.config.log_interval == 0 and index_batch > 0:
                cur_loss = sum(self.losses) / self.config.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, index_batch, len(dataloader), self.config.lr,
                    elapsed * 1000 / self.config.log_interval, cur_loss, math.exp(cur_loss)))
                self.losses = [0 for layer_index in range(self.config.num_hidden_layers)]
                start_time = time.time()

    def train(self, dataloader, epoch):
        """
        Args:
            - dataloader: torch.utils.DataLoader
        Returns:
            - 
        """
        optimizer = AdamW(
                    self.parameters(),
                    lr=float(self.config.learning_rate),
                    eps=float(self.config.adam_epsilon)
                )
        scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=len(dataloader)
            )
        start_time = time.time()
        #self.train()
        network_hidden_states, network_cell_states = self.init_hidden(self.config.num_hidden_layers, self.config.bsz, self.config.hidden_size)
        optimizer.zero_grad()
        for index_batch, (batch, ground_truth) in enumerate(dataloader):
            input_hidden_states = self.embedding(batch)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            #input_hidden_states = repackage_hidden(input_hidden_states)
            #network_hidden_states = [repackage_hidden(item) for item in network_hidden_states]
            #network_cell_states = [repackage_hidden(item) for item in network_cell_states]
            # attention à bien gérer les hidden-states dans le forward !
            for index_layer, (layer, loss, proj_head) in enumerate(zip(self.layers, self.loss_fcts, self.proj_heads)):
                hidden_states = [input_hidden_states, ] + network_hidden_states
                current_cell_state = network_cell_states[index_layer]
                current_hidden_state = network_hidden_states[index_layer]
                new_hidden, new_cell = layer(current_hidden_state, hidden_states, current_cell_state)
                network_hidden_states[index_layer] = new_hidden
                network_cell_states[index_layer] = new_cell
                if proj_head is not None:
                    predictions = proj_head(new_hidden)
                    loss_ = loss(predictions.view(-1, predictions.size(-1)), ground_truth.view(-1))
                    print(loss_)
                    #print(np.argmax(nn.Softmax(dim=-1)(predictions).detach().numpy(), axis=-1), ground_truth)
                    #if index_batch % 50 == 0 and index_batch > 0:
                    loss_.backward(retain_graph=True)
                    self.losses[index_layer] += loss_.item()

            nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)
            #loss_.backward()
            #for p in self.parameters():
            #    print(p.grad)
            
            optimizer.step()
            scheduler.step()
            if index_batch % 100 == 0 and index_batch > 0:
                optimizer.zero_grad()
            
            if index_batch % self.config.log_interval == 0 and index_batch > 0:
                cur_loss = sum(self.losses) / self.config.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, index_batch, len(dataloader), self.config.lr,
                    elapsed * 1000 / self.config.log_interval, cur_loss, math.exp(cur_loss)))
                self.losses = [0 for layer_index in range(self.config.num_hidden_layers)]
                start_time = time.time()
    
    def init_hidden(self, num_hidden_layers, bsz, hidden_size):
        """ Initialize to zeros the hidden state/cell.
        Arguments:
            - num_hidden_layers: int
            - bsz: int, batch size
            - hidden_size: int
        Returns:
            - torch.Variable (or tuple of torch.Variable)
        """
        factory_kwargs = {'device': self.config.device, 'dtype': self.config.dtype}
        hidden_states = [torch.nn.Parameter(torch.zeros((bsz, self.config.hidden_size), **factory_kwargs)) for i in range(num_hidden_layers)]
        cell_states = [torch.nn.Parameter(torch.zeros((bsz, self.config.hidden_size), **factory_kwargs)) for i in range(num_hidden_layers)]
        return hidden_states, cell_states


class Layer(nn.Module):
    """
    """
    def __init__(self, config, code=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        super().__init__()
        config = AttrDict(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon).double()
        self.attention = Attention(config).double()
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon).double()
        #self.c_fc = Conv1D(config.intermediate_size, config.hidden_size) # Conv1D(out_features, in_features)
        #self.act = ACT2FN[config.activation_function]
        #self.c_proj = Conv1D(config.hidden_size, config.intermediate_size) # Conv1D(out_features, in_features)
        self.dropout = nn.Dropout(config.dropout)
        self.lstm_cell = LSTMCell(config, code=code[4:])
    
    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, m):
        """Initialize the weights."""
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("AdaptiveEmbedding") != -1:
            if hasattr(m, "emb_projs"):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                self._init_weight(m.cluster_weight)
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                self._init_bias(m.cluster_bias)
            if hasattr(m, "out_projs"):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)
    
    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer):
                empty = torch.zeros(self.mem_len, bsz, self.config.d_model, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(self, layer_hidden_state, hidden_states, cell_state, mask=None):
        """
        """
        # We choose to put a LayerNorm before the attention like in GPT2 implementation and unlike BERT implementation,
        # because unlike BERT which is a stack of transformer layers with a LayerNorm at the end, we have LSTM units 
        # after the attention module.
        # dimension(layer_hidden_state): bsz_size, hidden_dim
        # dimension(hidden_states): bsz_size, hidden_dim
        # dimension(values): bsz_size, #tokens, hidden_dim  , where #tokens = #hidden_states + 1  (+1 for the embedding layer ot which we pay attention too)

        # Concatenating embedding input with network hidden-states
        bsz_size = layer_hidden_state.size(0)
        layer_hidden_state = layer_hidden_state.view(bsz_size, 1, -1)
        hidden_states = torch.cat([vector.view(bsz_size, 1, -1) for vector in hidden_states], dim=1)

        assert len(hidden_states.size())==3

        # Layer norm 1
        hidden_states_normed = self.ln_1(hidden_states)
        # Attention performed on all network current hidden-states
        attn_outputs = self.attention(layer_hidden_state, hidden_states_normed)
        # Skip-connection
        attn_output = attn_outputs[0] + layer_hidden_state
        # Layer norm 2
        hidden_state = self.ln_2(attn_output)
        # Dropout
        hidden_state = self.dropout(hidden_state)
        # Propagating hidden_states throught the LSTM cells
        output_dict = self.lstm_cell(hidden_state, cell_state) #{'hidden': hy, 'cell': cy, 'in': ingate, 'forget': forgetgate, 'out': outgate, 'c_tilde': cy_tilde}
        current_hidden = output_dict['hidden']
        current_cell = output_dict['cell']

        ## Projecting from hidden_size to intermediate_size
        #hidden_states = self.c_fc(hidden_states)
        ## Applying activation function
        #hidden_states = self.act(hidden_states)
        ## Projecting from intermediate_size to hidden_size
        #hidden_states = self.c_proj(hidden_states)
        ## Dropout
        #hidden_states = self.dropout(hidden_states)

        return current_hidden, current_cell


class Attention(nn.Module):
    """Attention module to perform attention on all current model hidden-states.
    """
    def __init__(self, config):
        """
        Args:
            - config: dict
        """
        super().__init__()
        config = AttrDict(config)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kv_attn = Conv1D(2 * self.hidden_size, self.hidden_size)
        self.q_attn = Conv1D(self.hidden_size, self.hidden_size)
        self.c_proj = Conv1D(self.hidden_size, self.hidden_size)
        self.proj_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.res_dropout = nn.Dropout(config.attention_res_dropout_prob)

        #self.max_positions = config.max_position_embeddings

    def _split_heads(self, x):
        """
        Splits hidden_size dimension into attention_head_size dim and num_attention_heads  
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # bsz_size, #heads, #tokens, head_dim

    def _merge_heads(self, tensor):
        """
        Merges attention_head_size dim and num_attention_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (self.num_attention_heads * self.attention_head_size,)
        return tensor.view(new_shape) # bsz_size, #tokens, #heads, hidden_size

    def forward(self, layer_hidden_state, hidden_states, output_attentions=False):
        """
        Args:
            - hidden_states: tensor (dimension(hidden_states) = bsz_size, #layer+1, hidden_size)
        Returns:
            - outputs: list
                - context_layer: tensor (dimension(context_layer) = bsz_size, #layer+1, hidden_size)
                - attention_probs: tensor(opt.) (dimension(attention_probs) = bsz_size, #layer+1, #layer+1)
        """
        # dimension(layer_hidden_state) = bsz_size, 1, hidden_size
        # dimension(hidden_states) = bsz_size, #tokens, hidden_size
        key_layer, value_layer = self.kv_attn(hidden_states).split(self.hidden_size, dim=2) # bsz_size, #layer+1, hidden_size
        query_layer = self.q_attn(layer_hidden_state) # bsz_size, 1, hidden_size
        query_layer = self._split_heads(query_layer) # bsz_size, #heads, 1, head_dim
        key_layer = self._split_heads(key_layer) # bsz_size, #heads, #tokens, head_dim
        value_layer = self._split_heads(value_layer) # bsz_size, #heads, #tokens, head_dim

        # dimension(key_layer.transpose(-1, -2)) = bsz_size, #heads, head_dim, #tokens
        # dimension(attention_scores) = bsz_size, #heads, 1, #tokens
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        ## For the future: beginning of code for incremental mask 
        #query_length, key_length = query_layer.size(-2), key_layer.size(-2)
        #causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        #attention_scores = torch.where(causal_mask, attention_scores, self.masked_bias.to(attention_scores.dtype))

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.proj_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        # Merging heads
        context_layer = self._merge_heads(context_layer)
        # dimension(context_layer) = bsz_size, 1, hidden_size
        context_layer = self.c_proj(context_layer)
        context_layer = self.res_dropout(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class LSTMCell(nn.Module):
    """
    """
    def __init__(self, config, code=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        super().__init__()
        config = AttrDict(config)
        gate_size = 4 * config.hidden_size
        factory_kwargs = {'device': config.device, 'dtype': config.dtype}

        self.hidden_size = config.hidden_size
        #self.weight_ih_l = torch.nn.Parameter(torch.empty((gate_size, self.hidden_size), **factory_kwargs))
        self.weight_hh_l = torch.nn.Parameter(torch.empty((gate_size, self.hidden_size), **factory_kwargs))
        #self.bias_ih_l = torch.nn.Parameter(torch.empty(gate_size, **factory_kwargs))
        self.bias_hh_l = torch.nn.Parameter(torch.empty(gate_size, **factory_kwargs))
        self.act_i = ACT2FN[config.activation_function_i]
        self.act_f = ACT2FN[config.activation_function_f]
        self.act_c = ACT2FN[config.activation_function_c]
        self.act_o = ACT2FN[config.activation_function_o]


        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset model parameters following uniform distribution between -1/sqrt(hidden_size) and 1/sqrt(hidden_size).
        """
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        self.weight_hh_l = torch.nn.init.uniform_(self.weight_hh_l, -stdv, stdv)
        self.bias_hh_l = torch.nn.init.uniform_(self.bias_hh_l, -stdv, stdv)

    def forward(self, hidden_state, cell_state):
        """
        Args:
            - hidden_state: torch.Tensor (dimension: hidden_size)
            - cell_state: torch.Tensor (dimension: hidden_size)
        """
        gates = F.linear(hidden_state, self.weight_hh_l, self.bias_hh_l)
        # dimension(gates) = bsz_size, , hidden_size

        ingate, forgetgate, cy_tilde, outgate = gates.chunk(4, -1) #dim modified: we split dimension 1 in 4 chunks of equal size

        ingate = self.act_i(ingate)
        forgetgate = self.act_f(forgetgate)
        cy_tilde = self.act_c(cy_tilde)
        outgate = self.act_o(outgate)

        cy = torch.mul(forgetgate, cell_state.view(cell_state.size(0), 1, -1)) + torch.mul(ingate, cy_tilde)
        hy = outgate * torch.tanh(cy)

        return {'hidden': hy, 'cell': cy, 'in': ingate, 'forget': forgetgate, 'out': outgate, 'c_tilde': cy_tilde}



def apply_mask(hidden_l, mask):
    if type(hidden_l) == torch.autograd.Variable:
        return hidden_l * mask
    else:
        return tuple(h * mask for h in hidden_l)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def create_dataloader_next_word_pred(input_path, context_size, bsz, train=False):
    """
    Args:
        - input_path: str (to pickle file)
        - context_size: int
        - bsz: int
        - train: bool
    Returns:
        - dataloader
    """
    if  isinstance(input_path, str) and os.path.exists(input_path):
        with open(input_path, 'rb') as inp:  # Overwrites any existing file.
            try:
                data = pickle.load(inp)
            except:
                data = pickle5.load(inp)
    else:
        data = input_path
    examples = [[item] for item in tqdm(data)]
    features = [torch.LongTensor(example).unsqueeze(0).to(torch.int64) for example in tqdm(examples)]
    input_ids = torch.cat(features, dim=0)
    labels_ids = input_ids[1:, :]
    input_ids = input_ids[:-1, :]
    data = TensorDataset(input_ids, labels_ids)
    sampler = RandomSampler(data) if train else None
    dataloader = DataLoader(data, sampler=sampler, batch_size=bsz)
    return dataloader

def create_examples(sequence, max_seq_length=512):
    """
    """
    return pad_to_max_length([50256] + sequence + [220, 50256], max_seq_length=max_seq_length)

def pad_to_max_length( sequence, max_seq_length=5):
    """Pad sequence to reach max_seq_length"""
    sequence = sequence[:max_seq_length]
    n = len(sequence)
    result = sequence + [220, 1] * ((max_seq_length - n)// 2)
    if len(result)==max_seq_length:
        return result
    else:
        return result + [220]

