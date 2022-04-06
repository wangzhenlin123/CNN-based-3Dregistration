import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class DCN(nn.Module):
    def __init__(self, input_channels,in_channels):
        super(DCN, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, in_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv3d(in_channels, in_channels*2, kernel_size=3, stride=2)
        self.conv3 = nn.Conv3d(in_channels*2, in_channels*4, kernel_size=3, stride=2)
        self.conv3_1 = nn.Conv3d(in_channels*4, in_channels*4, kernel_size=3, padding = "same")
        self.conv4 = nn.Conv3d(in_channels*4, in_channels*8, kernel_size=3, stride=2)
        self.conv5 = nn.Conv3d(in_channels*8, in_channels*16, kernel_size=3, stride=2)
        self.conv5_1 = nn.Conv3d(in_channels*16, in_channels*16, kernel_size=3, padding = "same")
        self.conv6 = nn.Conv3d(in_channels*16, in_channels*32, kernel_size=3, stride=2)
        self.conv6_1 = nn.Conv3d(in_channels*32, in_channels*32, kernel_size=3, padding = "same")
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, x, y,device):
        #Dense CNN
        input_value = torch.cat((x, y), 1)
        out = self.conv1(input_value)
        out = F.leaky_relu(out)

        out = self.conv2(out)
        out = F.leaky_relu(out)       

        out = self.conv3(out)
        out = F.leaky_relu(out) 
        
        out = self.conv3_1(out)
        out = F.leaky_relu(out)  

        out = self.conv4(out)
        out = F.leaky_relu(out)

        out = self.conv5(out)
        out = F.leaky_relu(out)  
        
        out = self.conv5_1(out)
        out = F.leaky_relu(out)
        
        out = self.conv6(out)
        out = F.leaky_relu(out)  
        
        out = self.conv6_1(out)
        out = F.leaky_relu(out)

        out = out.view(out.size(0), -1)
        
        #2 layers MLP
        out = self.fc1(out)
        out = F.leaky_relu(out)
        out = self.fc2(out)
        out = F.leaky_relu(out)
        
        
        #resgistration matrix
        tensor_0 = torch.zeros(out.shape[0],1).to(device)
        tensor_1 = torch.ones(out.shape[0],1).to(device)
        
        
        
        matrix1 = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0, out[:,-1:]]),
                torch.stack([tensor_0, torch.cos(out[:,:1]),-torch.sin(out[:,:1]),out[:,-2:-1]]),
                torch.stack([tensor_0,torch.sin(out[:,:1]),torch.cos(out[:,:1]),out[:,-3:-2]]),
                torch.stack([tensor_0,tensor_0,tensor_0,tensor_1])]).reshape(4,4,out.shape[0]).permute(2,0,1)

        matrix2 = torch.stack([
                torch.stack([torch.cos(out[:,1:2]),tensor_0,torch.sin(out[:,1:2]),tensor_0]),
                torch.stack([tensor_0,tensor_1,tensor_0,tensor_0]),
                torch.stack([-torch.sin(out[:,1:2]),tensor_0,torch.cos(out[:,1:2]),tensor_0]),
                torch.stack([tensor_0,tensor_0,tensor_0,tensor_1])]).reshape(4,4,out.shape[0]).permute(2,0,1)

        matrix3 = torch.stack([
                torch.stack([torch.cos(out[:,2:3]),-torch.sin(out[:,2:3]),tensor_0,tensor_0]),
                torch.stack([torch.sin(out[:,2:3]),torch.cos(out[:,2:3]),tensor_0,tensor_0]),
                torch.stack([tensor_0,tensor_0,tensor_1,tensor_0]),
                torch.stack([tensor_0,tensor_0,tensor_0,tensor_1])]).reshape(4,4,out.shape[0]).permute(2,0,1)

        final_matrix = torch.bmm(matrix1, matrix2)
        final_matrix = torch.bmm(final_matrix, matrix3)
        final_matrix = final_matrix[:,:-1, :]

        #spatial transformer
        grid = F.affine_grid(final_matrix, x.size())
        output = F.grid_sample(x, grid)

        return output,final_matrix





