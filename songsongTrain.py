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
		video=torch.from_numpy(video)
		hc1=myDecoder.initHidden(video)
		hc2=hc1
		outputs=[]
		betas=[]
		alphas=[]
		optimizer.zero_grad()
		for j, x in enumerate(video):
			output, hc1, hc2, beta, alpha=myDecoder(x, hc1, hc2)
			outputs.append(output)
			betas.append(beta)
			alphas.append(alpha)
		outputs=torch.stack(outputs,dim=0) #(seq_length, 1, nClasses)
		betas=torch.stack(betas,dim=0)     #(seq_length, 1)
		alphas=torch.stack(alphas, dim=0)  #(seq_length, 196, 1)
		outputs=torch.squeeze(outputs,dim=1) #(seq_length, nClasses)
		final=torch.mul(betas, outputs)      #(seq_length, nClasses)
		final=torch.sum(final, dim=0, keepdim=True)
		# print(final.size())
		print("label",label)
		print(label.size())
		loss = myLoss(final, label, alphas, betas)
		print("%d, loss:%f" % (i,loss.cpu().detach().numpy()))
		loss.backward()
		optimizer.step()

















if __name__== "__main__":
	main()