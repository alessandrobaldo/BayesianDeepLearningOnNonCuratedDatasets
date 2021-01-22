import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

from collections import OrderedDict
import tqdm
import random
import math

random.seed(42)

class NetClass(nn.Module):
	def __init__(self):
		super(NetClass, self).__init__()
	
		self.inputLayer = nn.Linear(1,50)
		self.hiddenLayer = nn.Linear(50,50)
		self.outputLayer = nn.Linear(50,1)
		self.tanh = nn.Tanh()
		self.act = nn.ReLU()
		
	def forward(self, x):
		x = self.inputLayer(x)
		x = self.act(x)
		x = self.hiddenLayer(x)
		x = self.act(x)
		x = self.hiddenLayer(x)
		x = self.act(x)
		x = self.outputLayer(x)
		x = self.tanh(x)
		x = (x+1)/2
		#x = F.log_softmax(x, dim=1)
		return x
		
def negative_log_prior(params):
	regularization_term = 0
	for name, W in params:
		regularization_term += W.norm(2)
	return 0.5*regularization_term



def runClass(Xtrain,ytrain,Xtest,learning_rate, epochs):
	# The nn package also contains definitions of popular loss functions; in this
	# case we will use Mean Squared Error (MSE) as our loss function.
	Xtrain,ytrain,Xtest = torch.from_numpy(Xtrain), torch.from_numpy(ytrain), torch.from_numpy(Xtest)
	Xtrain,Xtest, ytrain = Xtrain.type(torch.FloatTensor), Xtest.type(torch.FloatTensor), ytrain.type(torch.FloatTensor)
	#ytrain = ytrain.type(torch.LongTensor)
	
	model = NetClass()
	accs = []
	#loss_fn = torch.nn.NLLLoss()
	#loss_fn = torch.nn.CrossEntropyLoss()
	loss_fn = torch.nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	for t in range(epochs):
		model.train()
		y_pred = model(Xtrain)
		
		#y_pred = torch.max(y_pred, dim=1)[1].unsqueeze(1) #obtaining the classes
		#print(y_pred.size(), ytrain.size())
		loss = loss_fn(y_pred, torch.squeeze(ytrain)) + negative_log_prior(model.named_parameters())
		#if t % 100 == 99:
		#print(t, loss.item())
		
		model.zero_grad()
		optimizer.zero_grad()
		
		loss.backward()
		optimizer.step()
		'''
		with torch.no_grad():
			for param in self.model.parameters():
				param -= learning_rate * param.grad
		'''
		model.eval()
		y_pred = model(Xtest)
		y_pred = y_pred.detach().numpy()
		
	
	return y_pred