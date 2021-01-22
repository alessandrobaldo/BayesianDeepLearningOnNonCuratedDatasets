import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch

from collections import OrderedDict
import tqdm
import random
import math

random.seed(42)

class NetRegr(nn.Module):
	def __init__(self):
		super(NetRegr, self).__init__()
	
		self.inputLayer = nn.Linear(1,50)
		self.hiddenLayer = nn.Linear(50,50)
		self.outputLayer = nn.Linear(50,1)
		self.act = nn.ReLU()
	def forward(self, x):
		x = self.inputLayer(x)
		x = self.act(x)
		x = self.hiddenLayer(x)
		x = self.act(x)
		x = self.outputLayer(x)
		return x
		
def negative_log_prior(params):
	regularization_term = 0
	for name, W in params:
		regularization_term += W.norm(2)
	return 0.5*regularization_term

	
def runRegr(Xtrain,ytrain,Xtest,learning_rate, epochs):
	# The nn package also contains definitions of popular loss functions; in this
	# case we will use Mean Squared Error (MSE) as our loss function.
	Xtrain,ytrain,Xtest = torch.from_numpy(Xtrain), torch.from_numpy(ytrain), torch.from_numpy(Xtest)
	Xtrain,ytrain,Xtest = Xtrain.type(torch.FloatTensor), ytrain.type(torch.FloatTensor), Xtest.type(torch.FloatTensor)
	
	model = NetRegr()
	
	loss_fn = torch.nn.MSELoss(reduction='sum') #only likelihood term
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	#loss_fn = torch.nn.NLLLoss()
	for t in range(epochs):			
		# Forward pass: compute predicted y by passing x to the model. Module objects
		# override the __call__ operator so you can call them like functions. When
		# doing so you pass a Tensor of input data to the Module and it produces
		# a Tensor of output data.
		y_pred = model(Xtrain)
		# Compute and print loss. We pass Tensors containing the predicted and true
		# values of y, and the loss function returns a Tensor containing the
		# loss.
		loss = loss_fn(y_pred, ytrain) + negative_log_prior(model.named_parameters())
		#if t % 100 == 99:
		#print(t, loss.item())
		# Zero the gradients before running the backward pass.
		model.zero_grad()
		optimizer.zero_grad()
		# Backward pass: compute gradient of the loss with respect to all the learnable
		# parameters of the model. Internally, the parameters of each Module are stored
		# in Tensors with requires_grad=True, so this call will compute gradients for
		# all learnable parameters in the model.
		loss.backward()
		optimizer.step()

		# Update the weights using gradient descent. Each parameter is a Tensor, so
		# we can access its gradients like we did before.
		'''
		with torch.no_grad():
			for param in self.model.parameters():
				param -= learning_rate * param.grad
		'''
	return model(Xtest).detach().numpy()