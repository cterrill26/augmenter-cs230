import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

import torch
import math
import torch.nn as nn
from torch import Tensor

# %% Transformer model.
class PositionalEncoding(nn.Module):       
    def __init__(self, hidden_dim: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, hidden_dim, 2)* math.log(10000) / hidden_dim)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, hidden_dim))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, input: Tensor):
        return self.dropout(input + self.pos_embedding[:,:input.size(1), :])

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
            src_dim: int,
            trg_dim: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            hidden_dim: int,
            batch_first: bool,
            dim_feedforward: int = 2048,
            dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=hidden_dim,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first = batch_first)
        self.src_encoder = nn.Linear(src_dim, hidden_dim)
        self.trg_encoder = nn.Linear(trg_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        self.outs_decoder1 = nn.Linear(hidden_dim, dim_feedforward)
        self.outs_decoder2 = nn.Linear(dim_feedforward, trg_dim)


    def forward(self,
            src: Tensor,
            trg: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor):
        src_enc = self.positional_encoding(self.src_encoder(src))
        trg_enc = self.positional_encoding(self.trg_encoder(trg))
        outs = self.transformer(src_enc, trg_enc, src_mask, tgt_mask)
        outs = nn.functional.relu(self.outs_decoder1(outs))
        return self.outs_decoder2(outs)


# %% Dense model.
def get_dense_model(nFirstUnits, nHiddenUnits, nHiddenLayers, input_dim,
                    output_dim, L2_p, dropout_p, learning_r, loss_f,
                    marker_weights):
    
    # For reproducibility.
    np.random.seed(1)
    tf.random.set_seed(1)

    # Model.
    model = Sequential()
    # First layer.
    if L2_p > 0:
        model.add(Dense(nFirstUnits, input_shape=(input_dim,), 
                        kernel_initializer=glorot_normal(seed=None), 
                        activation='relu',
                        activity_regularizer=L2(L2_p)))
    else:
        model.add(Dense(nFirstUnits, input_shape=(input_dim,), 
                        kernel_initializer=glorot_normal(seed=None), 
                        activation='relu'))
    # Hidden layers.
    if nHiddenLayers > 0:
        for i in range(nHiddenLayers):
            if dropout_p > 0:
                model.add(Dropout(dropout_p))
            if L2_p > 0:
                model.add(Dense(nHiddenUnits, 
                                kernel_initializer=glorot_normal(seed=None), 
                                activation='relu',
                                kernel_regularizer=L2(L2_p)))            
            else:
                model.add(Dense(nHiddenUnits, 
                                kernel_initializer=glorot_normal(seed=None), 
                                activation='relu'))
    if dropout_p > 0:
        model.add(Dropout(dropout_p))
    # Last layer.
    model.add(Dense(output_dim, kernel_initializer=glorot_normal(seed=None), 
                    activation='linear'))
    
    # Optimizer.
    opt=Adam(learning_rate=learning_r)
    
    # Loss function.
    if loss_f == "weighted_mean_squared_error":
        model.compile(
            optimizer=opt,
            loss=weighted_mean_squared_error(marker_weights),
            metrics=[MeanSquaredError(), RootMeanSquaredError()])    
    else:
        model.compile(
            optimizer=opt,
            loss=loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return model

# %% LSTM model.
def get_lstm_model(input_dim, output_dim, nHiddenLayers, nHUnits, learning_r,
                   loss_f, bidirectional=False):
    
    # For reproducibility.
    np.random.seed(1)
    tf.random.set_seed(1)

    # Model.
    model = Sequential()
    # First layer.
    if bidirectional:
        model.add(Bidirectional(LSTM(units=nHUnits, 
                                     input_shape=(None, input_dim),
                                     return_sequences=True)))
    else:
        model.add(LSTM(units=nHUnits, input_shape = (None, input_dim),
                       return_sequences=True))
    # Hidden layers.
    if nHiddenLayers > 0:
        for i in range(nHiddenLayers):
            if bidirectional:
                model.add(Bidirectional(LSTM(units=nHUnits, 
                                             return_sequences=True)))
            else:
                model.add(LSTM(units=nHUnits, return_sequences=True))
    # Last layer.    
    model.add(TimeDistributed(Dense(output_dim, activation='linear')))
    
    # Optimizer.
    opt=Adam(learning_rate=learning_r)
    
    # Loss function.
    model.compile(
            optimizer=opt,
            loss=loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return model

# %% Helper functions.
def get_callback():
    callback =  tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-08, patience=10, verbose=0,
        mode='min', baseline=None, restore_best_weights=True )
    
    return callback

def weighted_mean_squared_error(weights):
    def loss(y_true, y_pred):      
        squared_difference = tf.square(y_true - y_pred)        
        weighted_squared_difference = weights * squared_difference  
        return tf.reduce_mean(weighted_squared_difference, axis=-1)
    return loss
