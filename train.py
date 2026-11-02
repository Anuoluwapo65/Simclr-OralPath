import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision 
import torch.utils.data as data 
from torch.utils.data import DataLoader
from torchvision import transforms 
import numpy as np 
import argparse
import os 
import json 
import random
import torch.optim as optim

import utils 
from torch.cuda.amp import autocast, GradScaler




model_name = ["resnet18", "resnet50", "resnet101"]

parser = argparse.ArgumentParser(description = "Train a pytorch SimCLR model on ORAL-CANCER DATASET")

parser.add_argument("-data", metavar = "DATASET", type = str,default=r"C:\Users\ACE\Downloads\Histopathological imaging database for Oral Cancer analysis\Histopathological imaging database for Oral Cancer analysis", help = "ORAL_CANCER dataset path")
parser.add_argument("-model", metavar = "MODEL", type = str, default = "resnet18", help = "Model to train")

parser.add_argument("-batch_size", metavar = "BATCH_SIZE", type = int, default = 256, help = "Batch size for training the dataset")
parser.add_argument("-epochs", metavar = "EPOCHS", type = int, default = 100, help = "Maximum number of epochs to be training the dataset")
parser.add_argument("-lr", metavar = "LEARNING_RATE", type = float, default = 0.001, help = "learning rate parameter for training the model")
parser.add_argument("-weight_decay", metavar = "WEIGHT_DECAY", type = float, default = 1e-4, help = "Weight decay parameter for the training the model")
parser.add_argument("-num_workers", metavar = "NUM_WORKERS", type = int, default = 4, help = "Number of workers for loading the dataset")
parser.add_argument("-device", metavar = "DEVICE", type = str, default = "cpu", help = "Device to train the model on the training dataset")
parser.add_argument("-temperature", metavar = "TEMPERATURE", type = float, default = 0.5, help = "Temperature parameter for the constrastive loss function")
parser.add_argument("-outdir", metavar = "OUTPUT_DIR", type = str, default = "outputs", help = "Directory for training outputs")


from model import ResNetSimCLR
from dataloader import TileLoader, ContrastiveLearningViewGenerator


def main():
    args = parser.parse_args()
    cpu = torch.device("cpu")
    cuda = torch.device("cuda") if torch.cuda.is_available() else cpu 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(42)
    args.device = cpu 

    os.makedirs(args.outdir, exist_ok = True)
    dataset = TileLoader(args.data)
    print("Images found:", len(dataset))
    print("Dataset path:", args.data)
    for index, image_path in enumerate(dataset.image_files):
        print(f"{index}: {image_path}")

    data = ContrastiveLearningViewGenerator(dataset, n_views = 2, transform = transforms.Compose([
        transforms.RandomResizedCrop(size = 224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p = 0.8),
        transforms.RandomGrayscale(p = 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]))
    print("Wrapped dataset:", len(data))
    print("Dataset size:", len(data))
    print("Dataset type:", type(data))

    train_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = ResNetSimCLR(base_model = args.model, out_dim = 128).to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0, last_epoch = -1)



    for epoch in range(args.epochs):
        model.train()
        lossmeter = utils.AverageMeter()
        acemeter = utils.AverageMeter()
        n_iter = 0 
        for image, _ in train_loader:
            image = torch.cat(image, dim = 0)
            image = image.to(args.device)

            with autocast(enabled = True):
                output = model(image)
                logits, labels = info_nce_loss(args, output)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


            lossmeter.update(loss.item(), image.size(0))
            acemeter.update(utils.accuracy(logits, labels).item(), logits.size(0))
            n_iter += 1

        print(
            f"Epoch [{epoch + 1} / {args.epochs}] "
            f"Loss: {lossmeter.avg:.4f} "
            f"Contrastive top-1 accuracy: {acemeter.avg:.2f}%"
        )
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best = False,
            checkpoint = args.outdir,
            filename = "checkpoint_path.tar",
        )




def info_nce_loss(args, output):
    # SimCLR InfoNCE loss implementation
    views = 2 
    labels = torch.cat([torch.arange(output.size(0) // views) for i in range(views)], dim = 0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(args.device)
    similarity_matrix = torch.matmul(output, output.T)
    # Create a mask to exclude the diagonal elements
    mask = torch.eye(labels.shape[0], dtype = torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim = 1)
    labels = torch.zeros(logits.shape[0], dtype = torch.long).to(args.device)
    logits = logits / args.temperature 
    return logits, labels 


if __name__ == "__main__":
    main()




