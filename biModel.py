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
        self.h1F=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.c1F=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.h2F=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.c2F=nn.Parameter(torch.randn(1, self.hiddenSize))

        self.h1B=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.c1B=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.h2B=nn.Parameter(torch.randn(1, self.hiddenSize))
        self.c2B=nn.Parameter(torch.randn(1, self.hiddenSize))

        self.spatialBiasF=nn.Parameter(torch.zeros(1, self.pixels))
        self.temporalBiasF=nn.Parameter(torch.zeros(1, 1))
        self.spatialBiasB=nn.Parameter(torch.zeros(1, self.pixels))
        self.temporalBiasB=nn.Parameter(torch.zeros(1, 1))

        self.h2pF=nn.Linear(self.hiddenSize, self.pixels)
        self.spatialC21F=nn.Linear(self.channels, 1)

        self.h2pB=nn.Linear(self.hiddenSize, self.pixels)
        self.spatialC21B=nn.Linear(self.channels, 1)

        self.h21F=nn.Linear(self.hiddenSize,1)
        self.temporalC21F=nn.Linear(self.channels,1)

        self.h21B=nn.Linear(self.hiddenSize,1)
        self.temporalC21B=nn.Linear(self.channels,1)

        self.lstm1F=nn.LSTMCell(self.channels, self.hiddenSize)
        self.lstm2F=nn.LSTMCell(self.hiddenSize,self.hiddenSize)

        self.lstm1B=nn.LSTMCell(self.channels, self.hiddenSize)
        self.lstm2B=nn.LSTMCell(self.hiddenSize,self.hiddenSize)

        
        self.fc=nn.Linear(self.hiddenSize*2,self.nClasses)

        self.relu=nn.ReLU()

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

        h1F=self.h1F.expand(self.batchSize, self.hiddenSize)
        c1F=self.c1F.expand(self.batchSize, self.hiddenSize)
        h2F=self.h2F.expand(self.batchSize, self.hiddenSize)
        c2F=self.c2F.expand(self.batchSize, self.hiddenSize)

        h1B=self.h1B.expand(self.batchSize, self.hiddenSize)
        c1B=self.c1B.expand(self.batchSize, self.hiddenSize)
        h2B=self.h2B.expand(self.batchSize, self.hiddenSize)
        c2B=self.c2B.expand(self.batchSize, self.hiddenSize)

        hc1F=(h1F,c1F)
        hc2F=(h2F,c2F)

        hc1B=(h1B,c1B)
        hc2B=(h2B,c2B)

        betasF=list()
        outputsF=list()
        alphasF=list()

        betasB=list()
        outputsB=list()
        alphasB=list()

        for step in range(videos.shape[0]):
            hiddenF=hc2F[0] #(N, hiddenSize)

            xF=videos[step].squeeze(0) #(N, channels, pixels)

            xF=xF.permute(0, 2, 1) #(N, pixels, channels)

            sAtten1F=self.h2pF(hiddenF) #(N, pixels)  spatial attention term1
            sAtten2F=self.spatialC21F(xF).squeeze(2) #(N, pixels) spatial attention term2

            energyF=torch.add(torch.add(sAtten1F, sAtten2F),
                            self.spatialBiasF.expand(self.batchSize, self.pixels) )#(batchSize, pixels)

            energyF=F.softmax(energyF,dim=1) #(N, pixels)

            #energy=F.normalize(energy,dim=1, p=1)

            alphasF.append(energyF)

            energyF=energyF.unsqueeze(1) #(N, 1, pixels)

            attn_appliedF=torch.bmm(energyF, xF) # (N, 1, channels)

            YF=attn_appliedF.squeeze(1) #(N, channels)

            betaF=self.h21F(hiddenF).squeeze(0)\
                    +self.temporalC21F(Y).squeeze(0)\
                    +self.temporalBiasF.expand(self.batchSize, 1)
            #beta=self.relu(beta)
            betasF.append(betaF)

            hc1F=self.lstm1(YF,hc1F)
            hc2F=self.lstm2(hc1F[0], hc2F)

            outputsF.append(hc2F[0])

        for step in range(videos.shape[0]-1, -1, -1):
            hiddenB=hc2B[0] #(N, hiddenSize)

            xB=videos[step].squeeze(0) #(N, channels, pixels)

            xB=xB.permute(0, 2, 1) #(N, pixels, channels)

            sAtten1B=self.h2pB(hiddenB) #(N, pixels)  spatial attention term1
            sAtten2B=self.spatialC21B(xB).squeeze(2) #(N, pixels) spatial attention term2

            energyB=torch.add(torch.add(sAtten1B, sAtten2B),
                            self.spatialBiasB.expand(self.batchSize, self.pixels) )#(batchSize, pixels)

            energyB=F.softmax(energyB,dim=1) #(N, pixels)

            #energy=F.normalize(energy,dim=1, p=1)

            alphasB.append(energyB)

            energyB=energyB.unsqueeze(1) #(N, 1, pixels)

            attn_appliedB=torch.bmm(energyB, xB) # (N, 1, channels)

            YB=attn_appliedB.squeeze(1) #(N, channels)

            betaB=self.h21B(hiddenB).squeeze(0)\
                    +self.temporalC21B(Y).squeeze(0)\
                    +self.temporalBiasB.expand(self.batchSize, 1)
            #beta=self.relu(beta)
            betasB.append(betaB)

            hc1B=self.lstm1(YB,hc1B)
            hc2B=self.lstm2(hc1B[0], hc2B)

            outputsB.append(hc2B[0])

        outputs=list()
        alphas=list()
        betas=list()

        outputs=[torch.cat(oF, oB) for oF,oB in zip(outputsF, outputsB)]
        outputs=[self.fc(o) for o in outputs]

        alphas=[torch.add(aF, aB) for aF, aB in zip(alphasF, alphasB)]
        betas=[torch.add(bF, bB) for bF, bB in zip(betasF, betasB)]

        alphasTensor=torch.stack(alphas, dim=0) #(seqLength, N, pixels)

        betasTensor=torch.stack(betas,dim=0)     #(seq_length, N, 1)
        betasTensor=betasTensor.permute(1, 2, 0)  #(N, 1, seqLength)
        betasTensor=F.softmax(betasTensor, dim=2)

        outputsTensor=torch.stack(outputs,dim=0) #(seq_length, N, nClasses)
        outputsTensor=outputsTensor.permute(1, 0, 2) #(N, seqLength, nClasses)

        logits=torch.bmm(betasTensor, outputsTensor).squeeze(1)      #(N, nClasses)

        return logits, alphasTensor, betasTensor


