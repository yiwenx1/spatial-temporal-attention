import torch.nn as nn
import random
import h5py
import torch.nn.functional as F
import torch
import pdb
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SongsongNet(nn.Module):
    def __init__(self, hiddenSize, nLayers, nClasses, seqLength):
        super(SongsongNet, self).__init__()
        self.hiddenSize=hiddenSize
        self.nLayers=nLayers
        self.nClasses=nClasses
        self.seqLength=seqLength

        self.flatImg=nn.Linear(196,1)

        self.lstm1=nn.LSTM(512, self.hiddenSize, self.nLayers)

        self.fc=nn.Linear(self.hiddenSize,self.nClasses)

        self.hidden2beta=nn.Linear(self.hiddenSize, self.seqLength)


    def forward(self, videos):
        # x: (N, frames, channels, pixels). (512,196)


        #videos=self.flatImg(videos)
        videos=torch.sum(videos, dim=3, keepdim=True)
        videos=videos.squeeze(3)
        videos=videos.permute(1,0,2);

        hidden=None

        output, hidden=self.lstm1(videos, hidden)

        output=output.permute(1,0,2) #(N, seqLength, hiddenSize)

        atten=hidden[0][-1]  #(N, hiddenSize)

        atten=self.hidden2beta(atten) #(N, seqLength)

        atten=atten.unsqueeze(1) #(N, 1, seqLength)

        logits=torch.bmm(atten, output) # (N, 1, hiddenSize)

        logits=logits.squeeze(1)

        return logits

