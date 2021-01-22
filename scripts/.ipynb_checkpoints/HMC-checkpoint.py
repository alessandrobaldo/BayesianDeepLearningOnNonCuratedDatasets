import random
from collections import OrderedDict
import tqdm
import math
import time
import torch

class HMC(object):
	def __init__(self,model,training, test, loss_fn,iterations):
		self.training = training
		self.test = test
		self.iterations = iterations
		self.m = 10
		
		#Model and Loss
		self.model = model
		self.loss_fn = loss_fn
		self.loss_prev = 0
		
		#Model Parameters
		self.state_dict = {}
		
		#Step Size
		self.step_size = 0.1
		
		#History
		self.history = {}
		
		#flag GPU
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
	
	'''NEGATIVE LOG PRIOR: REGULARIZATION TERM'''
	def negative_log_prior(self,params):
		regularization_term = 0
		for name, W in params:
			regularization_term += W.norm(2)
		return 0.5*regularization_term
		
	'''RANDOM INITIALIZATION OF THE MODEL''' 
	def initializeModel(self):
		self.state_dict = {}
		for name, param in self.model.named_parameters():
			size = list(param.size())
			mean = torch.zeros(size)
			std = torch.ones(size)
			new_param = torch.normal(mean, std) #initialising the params of each layer according to N(0,std)
			self.state_dict[name] = new_param
			self.state_dict[name].requires_grad = True
			self.state_dict[name] = self.state_dict[name].to(self.device)

		state_dict_zero = OrderedDict(self.state_dict)
		self.model.load_state_dict(state_dict_zero, strict=False) #loading the params on the model
		self.model = self.model.to(self.device)
	
	'''INITIALIZATION OF THE MOMENTUM AT THE BEGINNING OF EACH ITERATION'''
	def initializeMomentum(self):
		momentum = {}
		for name, param in self.model.named_parameters():
			size = list(param.size())
			mean = torch.zeros(size)
			std = torch.ones(size)
			new_param = torch.normal(mean, std) #initialising the params of each layer according to N(0,std)
			momentum[name] = new_param
			momentum[name] = momentum[name].to(self.device)
		return momentum
	
	
	'''HAMILTONIAN FUNCTION'''
	def hamiltonian(self, potential_energy, momentum):
		kinetic = 0
		for key, param in self.model.named_parameters():
			M = torch.ones(momentum[key].size()).cuda()
			kinetic += 0.5*torch.matmul(momentum[key].view(1,-1),torch.transpose(torch.mul(M, momentum[key]).view(1,-1),1,0))
		
		return torch.add(potential_energy, kinetic)
			
		
	'''FORWARD'''
	def forward(self,state_dict):
		state_dict_zero = OrderedDict(state_dict)
		self.model.load_state_dict(state_dict_zero, strict=False)
		for i, data in enumerate(self.training,0):
			images, labels = data
			images, labels = images.to(self.device), labels.to(self.device)
			y_pred = self.model(images, log_softmax=True)
		   
			loss = self.loss_fn(y_pred, labels) + self.negative_log_prior(self.model.named_parameters())
			break
		return loss
	
	'''MAIN'''
	def run(self):
		#1. random initialize model parameters
		self.initializeModel()
		#2. evaluate a first value of loss
		self.model.zero_grad()
		self.loss_prev = self.forward(self.state_dict)
		
		self.history = {
			"model_parameters":[self.state_dict],
			"losses": [self.loss_prev.item()]
		}
		
		'''OUTER LOOP'''
		for t in range(self.iterations):
			accepted = False
			start = time.time()
			
			prev_params = self.state_dict
			
			momentum = self.initializeMomentum() #evaluate momentum r = N(0, M)
			initialMomentum = momentum
			
			'''BACKPROPAGATING LOSS FOR 1st EVALUATION OF GRADEINTS'''
			self.loss_prev.backward(retain_graph = True)
			
			'''UPDATING MOMENTUM'''
			for key, param in self.model.named_parameters():
				momentum[key] = momentum[key] - 0.5*self.step_size*(param.grad)
				#update momentum: momentum = momentum - (step size)/2*SGD(potential energy) --> NLLLoss.backward()

			state_dict = self.state_dict 
			
			'''INNER LOOP'''
			for i in range(self.m):
				for key in self.state_dict.keys():
					'''UPDATING MODEL PARAMETERS'''
					M = torch.ones(momentum[key].size())
					M = M.to(self.device)
					#print(f"Param:{state_dict[key].size()} += M:{M.size()}*Momentum:{momentum[key].size()}")
					#state_dict[key] = state_dict[key] + self.step_size*torch.matmul(torch.inverse(M),momentum[key]) 
					#model parameters = model parameter - step size*inv(M)*momentum
					state_dict[key] = state_dict[key] + self.step_size*torch.mul(M,momentum[key])
				
				'''UPDATED MODEL PARAMETERS ---> FORWARD'''
				self.model.zero_grad()
				loss = self.forward(state_dict) #evaluate loss
				loss.backward(retain_graph = True)
				
				'''UPDATING MOMENTUM'''
				for key, param in self.model.named_parameters():
					momentum[key] = momentum[key] - self.step_size*(param.grad) #momentum = momentum - (step size)*SGD(potential energy)
			
			'''UPDATING MOMENTUM'''
			for key, param in self.model.named_parameters():
				momentum[key] = momentum[key] - 0.5*self.step_size*(param.grad)#momentum = momentum - (step size)/2*SGD(potential energy)
			
		   
			'''METROPOLIS HASTING CORRECTION'''
			r = torch.rand(1)
			prob = torch.exp(self.hamiltonian(loss,momentum)-self.hamiltonian(self.loss_prev,initialMomentum)).item()  
			#prob = exp(HamiltonianFunction(current parameters) - Hamiltionian(prev_parameters)) ---> Total Loss
			print(r, prob)
			if r < prob:
				accepted = True
				self.state_dict = state_dict
				self.loss_prev = loss
			end = time.time()
			
			'''SAVING ACCEPTED MODEL PARAMETERS AND LOSS'''
			self.history["model_parameters"].append(self.state_dict)
			self.history["losses"].append(self.loss_prev.item())
			print("Time {}/{}: {}s, Loss: {}, Minor Loss: {}, Accepted: {}".format(
				t+1,self.iterations, round(end-start,1),loss.item(),
				loss.item()<self.loss_prev.item(), accepted))
			
		return self.history