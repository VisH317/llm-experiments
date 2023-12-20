import torch
from torch import nn

class Lin(nn.Module):
    def __init__(self):
        super(Lin, self).__init__()
        self.lin = nn.Linear(1, 2)
    def forward(self, input):
        print("test")
        return self.lin(input)
    
lin1 = Lin()
lin2 = Lin()
lin3 = Lin()

ten = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
tens = ten.to_sparse()

print(tens.indices())