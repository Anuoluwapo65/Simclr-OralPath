import torch
import torch.nn as nn 
import torchvision.models as models 
from torchvision.models.resnet import ResNet, BasicBlock 


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained = False, num_classes = out_dim),
                            "resnet34": models.resnet34(pretrained = False, num_classes = out_dim)}

        self.backbone = self.resnet_dict[base_model]
        dim_mlp = self.backbone.fc.in_features 

        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)


    def forward(self, x):
        return self.backbone(x)



class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__(BasicBlock, [2, 2, 2,2])


    def push_forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
