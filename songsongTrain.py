from model import stDecoder
from loss import stAttentionLoss
import h5py
import torch
import numpy as np




def main():
	myDecoder=stDecoder(256,2,10)
	myLoss = stAttentionLoss(0.1, 0.01)
	optimizer=torch.optim.Adam(myDecoder.parameters(),lr=0.0001)
	videos=h5py.File("train.hdf5",'r')['train_set']
	labels=np.random.randint(10,size=20)
	labels=torch.LongTensor(labels)
	labels=torch.unsqueeze(labels, dim=1)
	# print(labels.size())
	for i,(video,label) in enumerate(zip(videos,labels)):
		optimizer.zero_grad()
		logits, alphas, betas=myDecoder(video)
		print("label",label)
		print(label.size())
		loss = myLoss(logit, label, alphas, betas)
		loss.backward()
		optimizer.step()

















if __name__== "__main__":
	main()
