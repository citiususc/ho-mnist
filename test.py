#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary
import numpy as np

from homnist.network import HONet, Net, HACCNet
from homnist.learning import MinMaxScale, test, test_hardware
from homnist.visualization import plot_data, plot_confusion_matrix

CURRENT_DIR = os.path.dirname(__file__)
MNIST_PATH = 'data'

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Hardware Oriented MNIST test')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-path', type=str, default="./models/mnist_quantized_converted.pth",
                        help='MNIST trained model after quantization and conversion')
    parser.add_argument('--plot-data', action='store_true', default=False,
                        help='For plotting the input data')
    parser.add_argument('--serial-port', type=str, default="/dev/ttyUSB2",
                        help='Serial port for connection to FPGA')
    parser.add_argument('--baudrate', type=int, default=1500000,
                        help='Baudrate for serial communication')
                        
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test-base', action='store_true', default=False,
                        help='Tests base network')
    group.add_argument('--test-quantized', action='store_true', default=False,
                        help='Tests quantized network')
    group.add_argument('--test-hardware', action='store_true', default=False,
                        help='Tests on real hardware')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        test_kwargs.update(cuda_kwargs)

    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([16, 16], interpolation=transforms.InterpolationMode.NEAREST),
        MinMaxScale()
        ])
    dataset_test = datasets.MNIST(os.path.join(CURRENT_DIR, MNIST_PATH), train=False, download=True, transform=transform_test)
    
    if args.plot_data:
        plot_data(dataset_test)
    
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    
    if args.test_base:
        model = Net(for_quantization=False)
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        summary(model, (1, 16, 16))
        gt_list, preds_list = test(model, device, test_loader)
        plot_confusion_matrix(gt_labels=gt_list, pred_labels=preds_list)
        
    elif args.test_quantized:
        from homnist.quantization import load_quantized_checkpoint
        model = Net(for_quantization=True)
        model = model.to(device)
        load_quantized_checkpoint(model, checkpoint_quantized=args.model_path, checkpoint_base=os.path.join(os.path.dirname(args.model_path), 'mnist_base.pth'))
        gt_list, preds_list = test(model, device, test_loader)
        plot_confusion_matrix(gt_labels=gt_list, pred_labels=preds_list)
    
    elif args.test_hardware:
        args.test_batch_size = 1
        test_kwargs = {'batch_size': args.test_batch_size}
        no_transform=transforms.Compose([transforms.ToTensor()])
        dataset_test_notransform = datasets.MNIST(os.path.join(CURRENT_DIR, MNIST_PATH), train=False, download=True, transform=no_transform)
        test_loader_notransform = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
        
        model = HACCNet(weights_dir=args.model_path, port=args.serial_port, baudrate=args.baudrate)
        gt_list, preds_list = test_hardware(model=model, test_loader=test_loader_notransform)
        plot_confusion_matrix(gt_labels=gt_list, pred_labels=preds_list)
        
    else:
        model = HONet()
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        summary(model, (1, 16, 16))
        gt_list, preds_list = test(model, device, test_loader)
        plot_confusion_matrix(gt_labels=gt_list, pred_labels=preds_list)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
