import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb


class stDecoder(nn.Module):
    def __init__(self, hiddenSize, nLayers, nClasses):
        super(stDecoder, self).__init__()
        self.hiddenSize=hiddenSize
        self.nLayers=nLayers
        self.nClasses=nClasses
        # self.initc=nn.Linear(512,self.hiddenSize)
        # self.inith=nn.Linear(512,self.hiddenSize)
        self.h1=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.c1=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.h2=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.c2=nn.Parameter(torch.randn(1, self.hiddenSize))

        self.sAtten=nn.Linear(self.hiddenSize+512, 1)
        self.tAtten=nn.Linear(self.hiddenSize, 1)
        self.lstm1=nn.LSTMCell(512, self.hiddenSize)
        self.lstm2=nn.LSTMCell(self.hiddenSize,self.hiddenSize)
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.fc=nn.Linear(self.hiddenSize,self.nClasses)

    # def initHidden(self,x):
    #     # x is the first flattend feature rectangel of a video
    #     x=x[0]
    #     #print(x.size())
    #     x=torch.mean(x,dim=1,keepdim=True) #(512, 1)
    #     x=torch.t(x)                       #(1, 512)
    #     c0=self.initc(x)                   #(1, hiddenSize)
    #     h0=self.inith(x)                   #(1, hiddenSize)
    #     return h0,c0


    def forward(self, video):
        # x: (frames, channels * pixels). (512,196)
        # hc1 is a (h,c) of the 1st hidden layer of LSTM
        # h is hideen state, c is cell state

        h1=self.h1
        c1=self.c1
        h2=self.h2
        c2=self.c2

        hc1=(h1,c1)
        hc2=(h2,c2)

        betas=list()
        outputs=list()
        alphas=list()

        for step in range(video.shape[0]):
            hidden=hc2[0]
            x=torch.t(video[step]) # 196*512

            beta=self.tAtten( hidden )  #(batchSize, 1)
            betas.append(beta.squeeze(0))

            hidden=hidden.view(-1)  # flatten hidden

            expandedHidden=torch.stack([hidden]*196, dim=0) #(196, hidden_size)

            sAttenInput=torch.cat((expandedHidden, x), dim=1)  # 196 x (hidden_size +512)

            sAttenInput=sAttenInput.unsqueeze(0) #(batchSize, 196, hidden_size+512)

            energy=self.sAtten(sAttenInput).squeeze(0)                   # (196, 1)

            attn_weights=F.softmax(energy,dim=0)            # (196, 1)

            alphas.append(attn_weights)

            attn_applied=torch.mul(attn_weights, x) # (196,512)

            Y=torch.sum(torch.t(attn_applied),dim=1) #(512,)

            Y=torch.unsqueeze(Y,0) #(batchSize,512)
            hc1=self.lstm1(Y,hc1)
            hc2=self.lstm2(hc1[0], hc2)

            output=self.tanh(hc2[0])
            outputs.append(self.fc(output))

        # pdb.set_trace()
        alphasTensor=torch.stack(alphas, dim=0)
        betasTensor=torch.stack(betas,dim=0)     #(seq_length, 1)
        outputsTensor=torch.stack(outputs,dim=0) #(seq_length, 1, nClasses)
        outputsTensor=outputsTensor.squeeze(1)

        logits=torch.mul(betasTensor, outputsTensor)      #(seq_length, nClasses)
        logits=torch.sum(logits, dim=0, keepdim=True)

        return logits, alphasTensor, betasTensor






##########################################

# myDecoder=stDecoder(256,2,10)

# video=torch.randn(20,512,196)

# logits, alphas, betas=myDecoder(video)
# pdb.set_trace()






