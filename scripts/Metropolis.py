import random
from collections import OrderedDict
import tqdm
import math
import time
import torch

random.seed(42)
class Metropolis(object):
    def __init__(self,model, training, loss_fn, stdev, iterations):
        self.training = training
        self.stdev = stdev
        self.iterations = iterations
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model
        
        self.loss_fn = loss_fn
        self.loss_prev = 0
        
        self.state_dict = {}
        
        self.acceptance_ratio = 0
        
        self.history = {}

    
    '''RANDOM INITIALIZATION OF THE MODEL'''
    def initializeModel(self):
        self.state_dict = {}
        for name, param in self.model.named_parameters():
            size = list(param.size())
            mean = torch.zeros(size)
            std = torch.ones(size) * self.stdev
            new_param = torch.normal(mean, std) #initialising the params of each layer according to N(0,std)
            self.state_dict[name] = new_param

        state_dict_zero = OrderedDict(self.state_dict)
        self.model.load_state_dict(state_dict_zero, strict=False) #loading the params on the model
        self.model = self.model.to(self.device)
    
    '''NEGATIVE LOG PRIOR: REGULARIZATION TERM'''
    def negative_log_prior(self,params):
        regularization_term = 0
        for name, W in params:
            regularization_term += W.norm(2)
        return 0.5*regularization_term
    
    '''FORWARD'''
    def forward(self):
        with torch.no_grad():
            loss = 0
            X, y = self.training
            X = X.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(X)
            #evaluating the first loss function --> will be used
            loss += self.loss_fn(y_pred, y) + self.negative_log_prior(self.model.named_parameters()) 
            return loss
    
    '''UPDATE OF MODEL PARAMETERS: param += N(0,stdev)'''
    def updateParameters(self):
        new_state_dict = {}
        for name, param in self.state_dict.items():
            size = list(param.size())
            mean = torch.zeros(size)
            std = torch.ones(size) * self.stdev
            new_param = param + torch.normal(mean, std)#creating the new set of parameters of the model, w_new = w_prev + N(0,std)
            new_state_dict[name] = new_param
        #setting the new set of params to the model, in order to test the new loss	
        state_dict_it = OrderedDict(new_state_dict)
        self.model.load_state_dict(state_dict_it, strict=False)
        return new_state_dict
    
    '''MAIN'''
    def run(self):
        self.initializeModel()
        self.loss_prev = self.forward()
        print("Initial Loss {}".format(self.loss_prev.item()))

        
        self.history = {
            "model_parameters":[self.state_dict],
            "losses": [self.loss_prev.item()],
            "acceptance_ratio": 0
        }
        
        for i in range(self.iterations):
            accepted = False
            start = time.time()
            new_state_dict = self.updateParameters()
            loss = self.forward()
            end = time.time()
            
            '''METROPOLIS PROCEDRE: ACCEPTANCE, CONDITIONAL ACCEPTANCE OR REJECTION'''
            if loss.item() < self.loss_prev.item():#direct acceptance
                self.state_dict = new_state_dict #set the set of params as the current accepted one
                self.loss_prev = loss #updating the loss
                self.acceptance_ratio +=1
                accepted = True
            else:
                r = torch.rand(1)
                prob = torch.exp(-loss-(-self.loss_prev))
                if r < prob: #conditional acceptance
                    self.state_dict = new_state_dict
                    self.loss_prev = loss
                    self.acceptance_ratio +=1
                    accepted = True
            
            '''SAVING ACCEPTED MODEL PARAMETERS AND LOSS'''
            self.history["model_parameters"].append(self.state_dict)
            self.history["losses"].append(self.loss_prev)
            self.history["acceptance_ratio"] = self.acceptance_ratio*100/(i+1)
            if (i+1)%(self.iterations/5)==0:
                print("Iteration {}/{}: {}s, Loss: {}, Minor Loss: {}, Accepted: {}, Current Acceptance Ratio: {}".format(
                i+1,self.iterations, round(end-start,1),loss.item(), loss.item()<self.loss_prev.item(), accepted,self.history["acceptance_ratio"]))
            
            state_dict_it = OrderedDict(self.state_dict)
            self.model.load_state_dict(state_dict_it, strict=False) 
            
        return self.history