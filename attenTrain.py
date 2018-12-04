from model import stDecoder
from loss import stAttentionLoss
import h5py
import torch
import numpy as np
import pdb
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


def main():
    myDecoder=stDecoder(256,2,51)
    myDecoder=myDecoder.to(DEVICE)

    print(myDecoder)
    myLoss = stAttentionLoss(0.1, 0.01)
    myLoss = myLoss.to(DEVICE)

    optimizer=torch.optim.Adam(myDecoder.parameters(),lr=0.001)
    videos=h5py.File("/home/ubuntu/train_feat.hdf5",'r')['train_feat']
    labels=h5py.File("/home/ubuntu/train_label.hdf5", 'r')['train_label']
    # print(labels.size())
    for epoch in epochs:

        begin=time.time()

        for i,(video,label) in enumerate(zip(videos,labels)):
            optimizer.zero_grad()

            video=torch.from_numpy(video)
            video=video.to(DEVICE)

            label=torch.tensor( np.array(label) ).long().unsqueeze(0)
            label=label.to(DEVICE)

            logits, alphas, betas=myDecoder(video)
            loss = myLoss(logits, label, alphas, betas)
            loss.backward()
            optimizer.step()
            if (i%20==0):
                print("video %d loss: %f" % (i, loss.cpu().detach().numpy()) )

        end=time.time()

        print("Epoch %d training time: %.2fs" %(epoch,(end-begin)) )
        fileName="params"+str(epoch)+".t7"                                                                         
        torch.save({'enParams':encoder.state_dict(), 'deParams':decoder.state_dict()},fileName)
        print("%s is saved." % fileName)

















if __name__== "__main__":
    main()
