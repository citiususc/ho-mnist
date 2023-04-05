from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ConvertToBlackWhite(object):
    def __init__(self, thresh=0.5, max_val=1.0):
        self.thresh = thresh
        self.max_val = max_val
        
    def __call__(self, tensor):
        bw_tensor = torch.zeros_like(tensor, requires_grad=False)
        bw_tensor[tensor >= self.thresh] = self.max_val

        return bw_tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(thresh={0}, max_val={1})'.format(self.thresh, self.max_val)


class MinMaxScale(object):
    def __init__(self, min_val=0.0, max_val=15.0, old_min_val=0.0, old_max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
        self.old_min_val = old_min_val
        self.old_max_val = old_max_val
        
    def __call__(self, tensor):
        tensor = tensor.to(torch.float32)
        
        tensor = (tensor - self.old_min_val) / (self.old_max_val - self.old_min_val) * (self.max_val - self.min_val) + self.min_val
        tensor = tensor.to(torch.int8)
        tensor = torch.clip(tensor, min=self.min_val, max=self.max_val)
        tensor = tensor.to(torch.float32)

        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1}, old_min_val={2}, old_max_val={3})'.format(self.min_val, self.max_val, self.old_min_val, self.old_max_val)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            if hasattr(args, 'dry_run') and args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pred_confidence = 0.0
    gt_confidence = 0.0
    loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    labels_list = []
    preds_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_criterion(output, target).item()
            output_preds = model.softmax(output)
            pred = output_preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred_confidence += np.array([output_preds[i, p].item() for i, p in enumerate(pred)]).sum().item()
            gt_confidence += np.array([output_preds[i, t].item() for i, t in enumerate(target)]).sum().item()
            labels_list.extend(target.tolist())
            preds_list.extend(pred.tolist())

    test_loss /= len(test_loader.dataset)
    pred_confidence /= len(test_loader.dataset)
    gt_confidence /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    print('\nAverage max confidence: {:.2f}%, Average correct confidence: {:.2f}%\n'.format(100. * pred_confidence, 100. * gt_confidence))
    
    return labels_list, preds_list

def test_hardware(model, test_loader):
    test_loss = 0
    correct = 0
    pred_confidence = 0.0
    gt_confidence = 0.0
    loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    labels_list = []
    preds_list = []
    for data, target in tqdm(test_loader):
        data = data.cpu().detach().numpy()  # Hardware model expects numpy arrays
        data = data[0, 0]  # We remove the batch and channel dimensions
        target = target[0]  # We remove the batch dimension
        
        output = model.forward(data)
        test_loss += loss_criterion(torch.from_numpy(output).unsqueeze(0), target.unsqueeze(0)).item()
        output_preds = model.softmax(output)
        pred = np.argmax(output_preds)  # get the index of the max log-probability
        correct += (pred == target)
        pred_confidence += output_preds[pred]
        gt_confidence += output_preds[target]
        labels_list.append(target)
        preds_list.append(pred)
        

    test_loss /= len(test_loader.dataset)
    pred_confidence /= len(test_loader.dataset)
    gt_confidence /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    print('\nAverage max confidence: {:.2f}%, Average correct confidence: {:.2f}%\n'.format(100. * pred_confidence, 100. * gt_confidence))
    
    return labels_list, preds_list


