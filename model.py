import torch.nn as nn
import torch.nn.functional as F
import torch


class stDecoder(nn.Module):
    def __init__(self, hiddenSize, nLayers, nClasses):
        super(stDecoder, self).__init__()
        self.hiddenSize=hiddenSize
        self.nLayers=nLayers
        self.nClasses=nClasses
        self.initc=nn.Linear(512,self.hiddenSize)
        self.inith=nn.Linear(512,self.hiddenSize)
        self.sAtten=nn.Linear(self.hiddenSize*self.nLayers+512, 1)
        self.tAtten=nn.Linear(self.hiddenSize*self.nLayers, 1)
        self.lstm1=nn.LSTMCell(512, self.hiddenSize)
        self.lstm2=nn.LSTMCell(self.hiddenSize,self.hiddenSize)
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.fc=nn.Linear(self.hiddenSize,self.nClasses)

    def initHidden(self,x):
        # x is the first flattend feature rectangel of a video
        x=torch.mean(x,dim=1,keepdim=True) #(512, 1)
        x=torch.t(x)                       #(1, 512)
        c0=self.initc(x)                   #(1, hiddenSize)
        h0=self.inith(x)                   #(1, hiddenSize)
        return h0,c0


    def forward(self, x, hc1, hc2):
        # x: (channels * pixels). (512,196)
        # hc1 is a (h,c) of the 1st hidden layer of LSTM
        # h is hideen state, c is cell state

        hidden=torch.cat((hc1[0],hc2[0]),dim=1) #(1, 2*hiddenSize)

        hidden=hidden.view(-1)  # flatten hidden 


        expandedHidden=torch.stack([hidden]*196, dim=0) #(196, num_layers* hidden_size)

        output=torch.cat((expandedHidden,torch.t(x)),dim=1)  # 196 x (num_layers* hidden_size +512)
        
        output=self.sAtten(output)                      # (196, 1)
        
        attn_weights=F.softmax(output,dim=0)            # (196, 1)
        
        attn_applied=torch.mul(attn_weights, torch.t(x)) # (196,512)

        Y=torch.sum(torch.t(attn_applied),dim=1) #(512,)

        Y=torch.unsqueeze(Y,0) #(1,512)
        
        hc1=self.lstm1(Y,hc1)
        hc2=self.lstm2(hc1[0], hc2)

        output=self.tanh(hc2[0])

        output=self.fc(output)

        hidden=torch.cat((hc1[0],hc2[0]),dim=1)
        beta=self.relu( self.tAtten(hidden))

        return output, hc1, hc2, beta






###############
myDecoder=stDecoder(256,2,10)

x=torch.randn(512,196)
oneVideo=[x]*10

hc1=myDecoder.initHidden(x)
hc2=hc1

outputs=[]
beta=[]

for i, x in enumerate(oneVideo):
    outputs[i], hc1, hc2, beta[i]=myDecoder(x, hc1, hc2)
