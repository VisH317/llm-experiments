import torch

tensor = torch.randn(4, 8, 4, 8)

mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).expand(4, 8)
print(tensor[tensor[0]!=0])