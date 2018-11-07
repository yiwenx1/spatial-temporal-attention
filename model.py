import torch.nn as nn
import torch.nn.functional as F
import torch


class stDecoder(nn.Module):
    def __init__(self, hiddenSize, hiddenLayers, nClasses):
        super(stDecoder, self).__init__()
        self.hiddenSize=hiddenSize
        self.hiddenLayers=hiddenLayers
        self.nClasses=nClasses
        self.sAtten=nn.Linear((self.hiddenSize*self.hiddenLayers+512),1)
        self.rnn=nn.LSTM(512,self.hiddenSize)
        

    def forward(self, x, hidden):
        # x: (channels * pixels). (512,196)
        # hidden: (num_layers, batch, hidden_size)

        hidden=torch.squeeze(hidden,1) #(num_layers , hidden_size)

        hidden=hidden.view(-1)

        expandedHidden=torch.stack([hidden]*196, dim=0) #(196, num_layers* hidden_size)

        output=torch.cat((expandedHidden,torch.t(x)),dim=1)  # 196 x (num_layers* hidden_size +512)
        
        output=self.sAtten(output)                      # (196, 1)
        
        attn_weights=F.softmax(output,dim=0)            # (196, 1)
        
        attn_applied=torch.mul(attn_weights, torch.t(x)) # (196,512)

        Y=torch.sum(torch.t(attn_applied),dim=1) #(512,)

        Y=torch.unsqueeze(Y,0)
        Y=torch.unsqueeze(Y,0)  #(1,1,512)
        
        print(Y.size())
        # self.rnn()



###############
myDecoder=stDecoder(256,2,10)
x=torch.randn(512,196)
hidden=torch.randn(2,256)

myDecoder(x,hidden)