from model import stDecoder
from loss import stAttentionLoss




def main():
	myDecoder=stDecoder(256,2,10)
	myLoss = stAttentionLoss(0.1, 0.01)
	optimizer=torch.optim.Adam(myDecoder.parameters(),lr=0.0001)
	for vidoe in videos:
		outputs=[]
		betas=[]
		alphas=[]
		optimizer.zero_grad()
		for i, x in enumerate(video):
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
		loss = myLoss(final, label, alphas, betas)
		loss.backward()
		optimizer.step()

















if __name__= "__main__":
	main()