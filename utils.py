import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
import torch.utils.data as data 
import numpy as np
import os 
import shutil


class GaussianBlur(object):
    def __init__(self, kernel_size, sigma = (0.1, 2.0)):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.kernel_size = kernel_size 
        self.sigma = sigma 
        self.bur_h = nn.Conv2d(3, 3, kernel_size = self.kernel_size, stride = 1, padding = radias, groups = 3, bias = False)
        self.bur_w = nn.Conv2d(3, 3, kernel_size = 3, stride = radias, padding = 1, groups = 3, bias = False)
        self.burr = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.bur_h, 
            self.bur_w
        )


    def __call__(self, sample):
        from PIL import ImageFilter 
        sample = sample.filter(ImageFilter.GaussianBlur(radius = np.random.uniform(self.sigma[0], self.sigma[1])))
        return sample 


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n = 2):
        self.val = val 
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count 



def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim = True)
    return correct_k.mul_(100.0 / batch_size)


def info_nce_loss(output, temperature):
    """Build SimCLR logits and positive-pair targets for a two-view batch."""
    if output.size(0) % 2 != 0:
        raise ValueError("InfoNCE expects an even batch containing two views per sample")

    output = F.normalize(output, dim=1)
    half_batch = output.size(0) // 2
    pair_ids = torch.arange(half_batch, device=output.device).repeat(2)
    positive_mask = pair_ids.unsqueeze(0) == pair_ids.unsqueeze(1)
    self_mask = torch.eye(output.size(0), dtype=torch.bool, device=output.device)
    candidate_mask = positive_mask & ~self_mask

    similarities = torch.matmul(output, output.T) / temperature
    positives = similarities[candidate_mask].view(output.size(0), 1)
    negatives = similarities[~positive_mask].view(output.size(0), -1)
    logits = torch.cat([positives, negatives], dim=1)
    targets = torch.zeros(output.size(0), dtype=torch.long, device=output.device)
    return logits, targets





def save_checkpoint(state, is_best, checkpoint = "checkpoint", filename = "checkpoint_path.tar"):
    os.makedirs(checkpoint, exist_ok = True)
    file_path = os.path.join(checkpoint, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(checkpoint, "model_best.pth.tar"))

        
                            
