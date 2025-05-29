import torch.nn as nn
from pytomography.projectors.SPECT import SPECTSystemMatrix

class Forward_Layer(nn.Module):
    def __init__(self, System_Matrix) -> None:
        super(Forward_Layer,self).__init__()
        self.System_Matrix = System_Matrix

    def forward(self, x):
        return self.System_Matrix.forward(x)
    
class Backward_Layer(nn.Module):
    def __init__(self, System_Matrix) -> None:
        super(Backward_Layer,self).__init__()
        self.System_Matrix = System_Matrix

    def forward(self, x):
        return self.System_Matrix.backward(x)