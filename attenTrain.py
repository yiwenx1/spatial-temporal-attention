import torch.nn as nn
import random
import h5py
import torch.nn.functional as F
import torch
import pdb
from model import stDecoder
from loss import stAttentionLoss
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    trainFeatures=h5py.File("./data/train_feat.hdf5",'r')["train_feat"]
    trainLabels=h5py.File("./data/train_label.hdf5", 'r')['train_label']
    print("Loading is done")
    print("trainFeatures.shape", trainFeatures.shape)
    print("trainLabels.shape", trainLabels.shape)

    batchSize=64
    model=stDecoder(batchSize, 512, 51)
    model=model.to(DEVICE)

    #criterion = stAttentionLoss(0.1, 0.01)
    criterion =nn.CrossEntropyLoss()
    criterion=criterion.to(DEVICE)

    optimizer=torch.optim.Adam(model.parameters())

    indexList=list(range(trainFeatures.shape[0]))
    batches=trainFeatures.shape[0]//batchSize
    epochs=20
    batchID=0

    for epoch in range(epochs):

        random.shuffle(indexList)
        begin=time.time()

        for j in range(batches):

            optimizer.zero_grad()

            batchIndexList=indexList[(j*batchSize):(j+1)*batchSize]
            batchIndexList.sort()

            videos=torch.from_numpy(trainFeatures[ batchIndexList ])
            videos=videos.to(DEVICE)

            labels=torch.from_numpy(trainLabels[ batchIndexList ]).long()
            labels=labels.to(DEVICE)

            logits, alphas, betas=model(videos)

            #print("alphas", alphas)
            print("betas", betas[0])
            #print("logits", logits)
            #print("labels", labels)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            batchID+=1
            if batchID%4==0 :
                print("batch %d loss is %f" %(batchID, loss.cpu().detach().numpy()))
                train_prediction = logits.cpu().detach().argmax(dim=1)
                train_accuracy = (train_prediction.numpy()==labels.cpu().numpy()).mean()
                print("train_accracy is %f" % train_accuracy)

        end=time.time()

        print("Epoch %d training time: %.2fs" %(epoch,(end-begin)) )
        #fileName="params"+str(epoch)+".t7"
        #torch.save(model.state_dict(),fileName)
        #print("%s is saved." % fileName)

if __name__ == '__main__':
    main()







