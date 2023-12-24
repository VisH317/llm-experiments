import torch

tensor = torch.randn(5, 2, 3, 4)

mask = torch.tensor([0, 1]).expand(5, 2)
print(mask)
expanded = mask.repeat(3, 4, 1, 1).permute(2, 3, 0, 1)
print(expanded.size())
print(tensor[expanded != 0].reshape(5, 1, 3, 4))
print(tensor[0][1])