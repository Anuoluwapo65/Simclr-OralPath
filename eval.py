import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as data 
import utils 
import argparse
from utils import accuracy, AverageMeter 


parser = argparse.ArgumentParser(description = "Evaluate a pytorch SIMCLR model on ORAL-CANCER DATASET")
parser.add_argument("-data", metavar = "DATASET", type = str, default = r"C:\Users\ACE\Downloads\Histopathological imaging database for Oral Cancer analysis\Histopathological imaging database for Oral Cancer analysis", help = "ORAL_CANCER dataset path")
parser.add_argument("-model", metavar = "MODEL", type = str, default = "resnet18", help = "Model to evaluate")
parser.add_argument("-batch_size", metavar = "BATCH_SIZE", type = int, default = 256, help = "Batch_size for evaluating the dataset")
parser.add_argument("-num_workers", metavar = "NUM_WORKERS", type = int, default = 4, help = "Number of workers for loading thr dataset")
parser.add_argument("-device", metavar = "DEVICE", type = str, default = "cpu", help = "Device to evaluate the model on the dataset")
parser.add_argument("-outdir", metavar = "OUTPUT_DIR", type = str, default = "outputs", help = "Directory for evaluation outputs")


from model import ResNetSimCLR
from torchvision import models
import os 
import json 
import torch.utils.data as data 
import numpy as np 
from dataloader import TileLoader, ContrastiveLearningViewGenerator 
from utils import AverageMeter 
from torch.utils.data import DataLoader 
import sklearn 
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
from openslide import OpenSlide 
from PIL import Image
import matplotlib
 
matplotlib.use("Agg")

def main():
    args = parser.parse_args()
    cpu = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else cpu 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.enabled = False
    args.device = device 
    args.outdir = os.path.join(args.outdir, "eval")
    args.cpu = cpu 
    torch.manual_seed(42)


    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok = True)



    dataset = TileLoader(args.data, transform = ContrastiveLearningViewGenerator())

    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    model = models.resnet18(pretrained = False, num_classes = 128).to(args.device)

    checkpoint = torch.load(os.path.join(args.outdir, "checkpoint_path.tar"), map_location = args.device)
    model.load_state_dict(checkpoint["state_dict"])

    for k in model.state_dict().keys():
        print(k)

    model.state_dict(state_dict = checkpoint["state_dict"])

    model.fc = nn.Identity()
    model = nn.DataParallel(model)


    embeddings = torch.zeros(len(dataloader.dataset), 128).to(args.device)
    model.eval()
    with torch.no_grad():
        lossmeter = utils.AverageMeter()
        acemeter = utils.AverageMeter()

        for i, (image, _) in enumerate(dataloader):
            image = image.to(args.device)
            output = model(image)
            embeddings[i * args.batch_size:(i + 1) * args.batch_size] = output.detach().cpu()
            loss = F.cross_entropy(output, torch.zeros(output.size(0), dtype = torch.long).to(args.device))
            accuracy = utils.accuracy(output, torch.zeros(output.size(0), dtype = torch.long).to(args.device))
            lossmeter.update(loss.item(), image.size(0))
            acemeter.update(accuracy.item(), image.size(0))
            acemeter.update(accuracy.item(), image.size(0))

        print(f"Evaluation loss: {lossmeter.avg:.4f}, Evaluation accuracy: {acemeter.avg:.4f}")



class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        return x 



if __name__ == "__main__":
    main()






