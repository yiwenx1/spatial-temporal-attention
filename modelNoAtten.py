import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SongsongNet(nn.Module):
    def __init__(self, hiddenSize, nLayers, nClasses, seqLength):
        super(stDecoder, self).__init__()
        self.hiddenSize=hiddenSize
        self.nLayers=nLayers
        self.nClasses=nClasses
        self.seqLength=seqLength

        self.flatImg=nn.Linear(196,1)

        self.lstm1=nn.LSTM(512, self.hiddenSize, self.nLayers)

        self.fc=nn.Linear(self.hiddenSize,self.nClasses)

        self.hidden2beta(self.hiddenSize, self.seqLength)


    def forward(self, videos):
        # x: (N, frames, channels, pixels). (512,196)


        videos=self.flatImg(videos)
        videos=videos.squeeze(3)
        videos=videos.permute(1,0,2);

        hidden=None

        output, hidden=self.lstm1(videos, hidden)

        output.permute(1,0,2) #(N, seqLength, hiddenSize)

        atten=hidden[0][-1]  #(N, hiddenSize)

        atten=self.hidden2beta(beta) #(N, seqLength)

        atten=atten.unsqueeze(1) #(N, 1, seqLength)

        logits=torch.bmm(atten, output) # (N, 1, hiddenSize)

        logits.squeeze(1)

        return logits


def main():
    trainFeatures=h5py.File("/home/ubuntu/train_feat.hdf5",'r')['train_feat']
    trainLabels=h5py.File("/home/ubuntu/train_label.hdf5", 'r')['train_label']
    print("Loading is done")

    model=SongsongNet(512, 2, 51, 20)
    model=model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to(DEVICE)

    optimizer=torch.optim.Adam(model.parameters())

    batchSize=64
    indexList=list(range(trainFeatures.shape[0]))
    batches=trainFeatures.shape[0]//batchSize
    epochs=20
    batchID=0

    for epoch in range(epochs):

        random.shuffle(indexList)
        begin=time.time()

        for j in range(batches):

            optimizer.zero_grad()

            videos=torch.from_numpy(trainFeatures[(j*batchSize):(j+1)*batchSize])
            videos=videos.to(DEVICE)

            labels=torch.from_numpy(trainLabels[(j*batchSize):(j+1)*batchSize])
            labels=labels.to(DEVICE)

            logits=model(videos)

            loss=criterion(logits, labels)

            loss.backward()

            optimizer.step()

            batchID+=1
            if batchID%20==0 :
                print("batch %d loss is %f" %(batchID, loss.cpu().detach().numpy()))
                train_prediction = output.cpu().detach().argmax(dim=1)
                train_accuracy = (train_prediction.numpy()==labels.cpu().numpy()).mean()
        
        end=time.time()

        print("Epoch %d training time: %.2fs" %(epoch,(end-begin)) )
        fileName="params"+str(epoch)+".t7"                                                                         
        torch.save(model.state_dict(),fileName)
        print("%s is saved." % fileName)



if __name__ == '__main__':
    main()






