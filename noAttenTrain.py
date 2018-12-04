import torch.nn as nn
import random
import h5py
import torch.nn.functional as F
import torch
import pdb
from modelNoAtten import SongsongNet
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    trainFeatures=h5py.File("./data/train_128.hdf5",'r')["train_128"]
    trainLabels=h5py.File("./data/train_label.hdf5", 'r')['train_label']
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

            labels=torch.from_numpy(trainLabels[(j*batchSize):(j+1)*batchSize]).long()
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






