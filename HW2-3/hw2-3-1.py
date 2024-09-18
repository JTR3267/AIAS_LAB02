import torch
import torch.nn as nn

class Subgraph1(nn.Module):
    def __init__(self):
        super(Subgraph1, self).__init__()
    
    def forward(self, input1, input2):
        return torch.matmul(input1, input2)

input1 = torch.randn(128, 128)
input2 = torch.randn(128, 128)
subgraph1 = Subgraph1()
torch.onnx.export(subgraph1, (input1, input2), 'subgraph1.onnx', input_names=['A', 'B'], output_names=['C'])