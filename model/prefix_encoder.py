""" Utility classes and functions related to SwitchPrompt (EACL 2023).
Copyright (c) 2022 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch
from torch.autograd import Variable
import torch.nn as nn

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.size = config.num_hidden_layers * 2 * config.hidden_size
        self.pre_seq_len = config.pre_seq_len
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding1 = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)
            self.embedding2 = torch.nn.Embedding(9, config.num_hidden_layers * 2 * config.hidden_size)

            self.lstm_head = torch.nn.LSTM(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )

            self.new_layer = torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            self.gate1 = torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            self.gate2 = torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            self.weight1 =nn.Parameter(torch.FloatTensor(1,config.num_hidden_layers * 2 * config.hidden_size), requires_grad=True)
            self.weight2 = nn.Parameter(torch.FloatTensor(1,config.num_hidden_layers * 2 * config.hidden_size), requires_grad=True)
            #initialization of weights in the range of xaviers normal distribution
            torch.nn.init.xavier_normal_(self.weight1) #torch.nn.init.xavier_normal_
            torch.nn.init.xavier_normal_(self.weight2)
            torch.nn.init.xavier_normal_(self.new_layer.weight)
            torch.nn.init.xavier_normal_(self.gate1.weight)
            #torch.nn.init.xavier_normal_(self.embedding1.weight)

    def forward(self,context_word,pooled_output1,device,batch_size, prefix: torch.Tensor,prefix1: torch.Tensor,):
        if self.prefix_projection:
            prefix_tokens1 = self.embedding1(prefix)
            prefix_tokens2 = self.embedding2(prefix)
            past_key_values = prefix_tokens1 + prefix_tokens2
            past_key_values = torch.nn.Sigmoid(past_key_values)
        else:
            prefix_tokens1 = self.embedding1(prefix)
            m = torch.nn.Sigmoid()
            context_word = m(self.new_layer(context_word))
            batch_size,word_size,embedding_dimension = context_word.shape
            #keyword as prefix and suffix
            padded_a = torch.cat([context_word,prefix_tokens1], dim = 1)
            padded_b = torch.cat([prefix_tokens1, context_word], dim = 1)
            batch_size,length,embedding = padded_a.shape
            #calculation of gates
            self.weight1 = self.weight1.to(device)
            self.weight2 = self.weight2.to(device)
            pooled_output1 = pooled_output1.unsqueeze(1)
            pooled_output1 = pooled_output1.repeat(1,length,1)
            gate1 = m(self.gate1(pooled_output1))
            w1 = (self.weight1)
            w2 = (self.weight2)
            m = torch.nn.Sigmoid()
            gate1 = m((w1 * gate1))
            gate2 = m((w2 * gate1))
            #padding to s = (m-n)
            zeroes = torch.zeros(batch_size, (length-self.pre_seq_len) , self.size).long().to(device)
            padded_prefix = torch.cat([prefix_tokens1,zeroes], dim = 1)
            #switchprompt calculation
            past_key_values  =  (gate1 * (padded_prefix)) +   ((1 - (gate1)) * (gate2 * (padded_a) + ((1 - (gate2)) *  (padded_b))))
            batch_size,seq_size,embedding_dimension = past_key_values.shape
            
        return past_key_values,seq_size