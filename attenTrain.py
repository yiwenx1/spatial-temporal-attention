import torch.nn as nn
import random
import h5py
import torch.nn.functional as F
import torch
import pdb
from tensorboard_logging import Logger
from model import stDecoder
from loss import stAttentionLoss
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def val(model, valFeatures, valLabels, batchSize):
    model.eval()
    batches=valFeatures.shape[0]//batchSize
    acc=0
    for i in range(batches):

        videos=valFeatures[ i*batchSize : (i+1)*batchSize ]
        videos=torch.from_numpy(videos)
        videos=videos.to(DEVICE)

        labels=valLabels[ i*batchSize : (i+1)*batchSize ]
        labels=torch.from_numpy(labels).long()
        labels=labels.to(DEVICE)

        logits,_ ,_=model(videos)
        prediction = logits.cpu().detach().argmax(dim=1)
        accuracy = (prediction.numpy()==labels.cpu().numpy()).mean()
        acc+=accuracy
    print("Val accuracy: ", acc/batches)


def main():
    tLog = Logger("./logs")
    trainFeatures=h5py.File("./data/train_feat.hdf5",'r')["train_feat"]
    trainLabels=h5py.File("./data/train_label.hdf5", 'r')['train_label']
    valFeatures=h5py.File("./data/val_feat_v2.hdf5",'r')['val_feat']
    valLabels=h5py.File("./data/val_label.hdf5", 'r')['val_label']
    print("Loading is done")
    print("trainFeatures.shape", trainFeatures.shape)
    print("trainLabels.shape", trainLabels.shape)
    print("valFeatures.shape", valFeatures.shape)
    print("valLabels.shape", valLabels.shape)

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

    val(model, valFeatures, valLabels, batchSize)
    for epoch in range(epochs):

        random.shuffle(indexList)
        begin=time.time()

        for j in range(batches):
            model.train()

            optimizer.zero_grad()

            batchIndexList=indexList[(j*batchSize):(j+1)*batchSize]
            batchIndexList.sort()

            videos=torch.from_numpy(trainFeatures[ batchIndexList ])
            videos=videos.to(DEVICE)

            labels=torch.from_numpy(trainLabels[ batchIndexList ]).long()
            labels=labels.to(DEVICE)

            logits, alphas, betas=model(videos)

            #print("alphas", alphas)
            #print("betas", betas[0])
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
                tr_info = { 'loss': loss.cpu().detach().numpy(), 'accuracy': train_accuracy }
                for tag, value in tr_info.items():
                    tLog.log_scalar(tag, value, batchID+1)

        end=time.time()

        print("Epoch %d training time: %.2fs" %(epoch,(end-begin)) )
        val(model, valFeatures, valLabels, batchSize)
        #fileName="params"+str(epoch)+".t7"
        #torch.save(model.state_dict(),fileName)
        #print("%s is saved." % fileName)

if __name__ == '__main__':
    main()







