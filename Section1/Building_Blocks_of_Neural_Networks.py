import torch
import torch.nn as nn
import torch.nn.functional as F


#nn.Sequential
My_neuralnet = nn.Sequential(
    nn.Linear(3,2),
    nn.ReLU(),
    nn.Linear(2,1),
    nn.Sigmoid()

)

#nn.Module Inheritance

class MyNeuralNet(nn.Module):
    def __init__(self,input_size,n_nodes,output_size):
        super(MyNeuralNet,self).__init__()
        self.operationOne = nn.Linear(input_size,n_nodes)
        self.operationTwo = nn.Linear(n_nodes,output_size)

    def forward(self,x):
        x = F.relu(self.operationOne(x))
        x = self.operationTwo(x)
        x = F.sigmoid(x)
        return x


