import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb


class stDecoder(nn.Module):
    def __init__(self, batchSize, hiddenSize, nClasses):
        super(stDecoder, self).__init__()
        self.batchSize=batchSize
        self.hiddenSize=hiddenSize
        self.nClasses=nClasses
        self.pixels=196
        self.channels=512
        self.h1=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.c1=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.h2=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.c2=nn.Parameter(torch.randn(1, self.hiddenSize))

        self.spatialBias=nn.Parameter(torch.zeros(1, self.pixels))
        self.temporalBias=nn.Parameter(torch.zeros(1, 1))

        self.h2p=nn.Linear(self.hiddenSize, self.pixels)
        self.spatialC21=nn.Linear(self.channels, 1)

        self.h21=nn.Linear(self.hiddenSize,1)
        self.temporalC21=nn.Linear(self.channels,1)

        self.lstm1=nn.LSTMCell(self.channels, self.hiddenSize)
        self.lstm2=nn.LSTMCell(self.hiddenSize,self.hiddenSize)
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


    def forward(self, videos):
        # videos: (N, frames, channels, pixels)
        videos=videos.permute(1,0,2,3) #(frames, N, channels, pixels)

        h1=self.h1.expand(self.batchSize, self.hiddenSize)
        c1=self.c1.expand(self.batchSize, self.hiddenSize)
        h2=self.h2.expand(self.batchSize, self.hiddenSize)
        c2=self.c2.expand(self.batchSize, self.hiddenSize)

        hc1=(h1,c1)
        hc2=(h2,c2)

        betas=list()
        outputs=list()
        alphas=list()

        for step in range(videos.shape[0]):
            hidden=hc2[0] #(N, hiddenSize)

            x=videos[step].squeeze(0) #(N, channels, pixels)

            x=x.permute(0, 2, 1) #(N, pixels, channels)

            sAtten1=self.h2p(hidden) #(N, pixels)  spatial attention term1
            sAtten2=self.spatialC21(x).squeeze(2) #(N, pixels) spatial attention term2

            energy=torch.add(torch.add(sAtten1, sAtten2),
                            self.spatialBias.expand(self.batchSize, self.pixels) )#(batchSize, pixels)

            energy=F.softmax(energy,dim=1) #(N, pixels)

            #energy=F.normalize(energy,dim=1, p=1)

            alphas.append(energy)

            energy=energy.unsqueeze(1) #(N, 1, pixels)

            attn_applied=torch.bmm(energy, x) # (N, 1, channels)

            Y=attn_applied.squeeze(1) #(N, channels)

            beta=self.h21(hidden).squeeze(0)\
                    +self.temporalC21(Y).squeeze(0)\
                    +self.temporalBias.expand(self.batchSize, 1)
            #beta=self.relu(beta)
            betas.append(beta)

            hc1=self.lstm1(Y,hc1)
            hc2=self.lstm2(hc1[0], hc2)

            output=self.fc(hc2[0]) #(N, nClasses)
            outputs.append(output)

        alphasTensor=torch.stack(alphas, dim=0) #(seqLength, N, pixels)

        betasTensor=torch.stack(betas,dim=0)     #(seq_length, N, 1)
        betasTensor=betasTensor.permute(1, 2, 0)  #(N, 1, seqLength)
        betasTensor=F.softmax(betasTensor, dim=2)

        outputsTensor=torch.stack(outputs,dim=0) #(seq_length, N, nClasses)
        outputsTensor=outputsTensor.permute(1, 0, 2) #(N, seqLength, nClasses)

        logits=torch.bmm(betasTensor, outputsTensor).squeeze(1)      #(N, nClasses)

        return logits, alphasTensor, betasTensor


