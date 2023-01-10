import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
########################### note ############################
# torch 0.4: https://pytorch.org/2018/04/22/0_4_0-migration-guide.html
# best sound avg pooling is 200(filter num) * 8(pool)
#############################################################

class LSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim , type_num = 10):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim)
        self.out = nn.Linear(64, 10)
        # self.hidden = self.init_hidden()
        
    # def init_hidden(self):
    #     return(Variable(torch.zeros(4,1,self.hidden_dim),Variable(torch.zeros(1,1,self.hidden_dim))))

    def forward(self,x1, x2, hid):
        h_c1, h_c2, h_n1, h_n2 = hid 
        r_out1,(h_n1,h_c1) = self.lstm1(x1,(h_n1,(h_c1 + h_c2))) # None for all zero initial hidden state
        r_out2,(h_n2,h_c2) = self.lstm2(x2,(h_n2,(h_c1 + h_c2))) # None for all zero initial hidden state
        r_out = (r_out1 + r_out2)/2
        # r_out = r_out.view(r_out.size(0)*self.hidden_dim,-1)
        out = self.out(r_out[:,-1,:])        
        return out, [h_c1, h_c2, h_n1, h_n2]

class LSTMCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def _init_hidden(self, input_):
        h = torch.zeros_like(input_.view(1, input_.size(1), -1))
        c = torch.zeros_like(input_.view(1, input_.size(1), -1))
        return h, c
    def forward(self, x, hidden):
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)
    
        # gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, torch.tanh(cy))
        
        return (hy, cy)
class LSTMModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, half_hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim
               
        self.lstm1 = LSTMCell(input_dim1, hidden_dim, layer_dim)
        self.lstm2 = LSTMCell(input_dim2, hidden_dim, layer_dim)
        self.fc1 = nn.Linear(hidden_dim, half_hidden_dim)
        self.fc2 = nn.Linear(half_hidden_dim, output_dim)
    
    
    def forward(self, x1, x2, score, device):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim)).to(device)
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim)).to(device)
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim))

                    
       
        outs = []
        
        cn1 = c0[0,:,:]
        hn1 = h0[0,:,:]
        cn2 = c0[0,:,:]
        hn2 = h0[0,:,:]

        for seq in range(len(x1)):
            # print(x1.size(0))
            if seq == 0:
                weight = 1
            else:
                weight = 2 * score[seq-1]
            cn_sum_for_lstm1 = (cn1 * (1 / (1 + weight)) + cn2 * (weight / (1 + weight))).to(device)
            cn_sum_for_lstm2 = (cn2 * (1 / (1 + weight)) + cn1 * (weight / (1 + weight))).to(device)
            hn1, cn1 = self.lstm1(x1[seq], (hn1, cn_sum_for_lstm1))
            hn2, cn2 = self.lstm2(x2[seq], (hn2, cn_sum_for_lstm2))
            outs.append(hn1[0]+hn2[0])
        outs_tensor = torch.cat(outs[:]).view(len(outs),-1)  
        # out = outs[-1]
        # print(out)
        out1 = self.fc1(outs_tensor).to(device)
        out2 = self.fc2(out1).to(device)
        out = F.sigmoid(out2)
        return out
