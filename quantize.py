#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from homnist.network import Net
from homnist.learning import train, test, ConvertToBlackWhite, MinMaxScale, AddGaussianNoise
from homnist.quantization import get_quant

CURRENT_DIR = os.path.dirname(__file__)
MNIST_PATH = 'data'

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Quantization of Hardware Oriented MNIST CNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-path', type=str, default="./models/mnist_base.pth",
                        help='MNIST base float32 model')        
    parser.add_argument('--output-path', type=str, default="./models/mnist_quantized.pth",
                        help='MNIST quantized model')
    parser.add_argument('--plot-data', action='store_true', default=False,
                        help='For plotting the input data')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([ConvertToBlackWhite()], p=0.5),
        transforms.RandomApply([AddGaussianNoise(0.0, 0.1)], p=0.5),
        transforms.RandomResizedCrop([16, 16], scale=(0.8, 1.1), interpolation=transforms.InterpolationMode.NEAREST),
        MinMaxScale()
        ])
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([16, 16], interpolation=transforms.InterpolationMode.NEAREST),
        MinMaxScale()
        ])
    dataset_train = datasets.MNIST(os.path.join(CURRENT_DIR, MNIST_PATH), train=True, download=True,
                       transform=transform)
    dataset_test = datasets.MNIST(os.path.join(CURRENT_DIR, MNIST_PATH), train=False, download=True,
                       transform=transform_test)
    
    if args.plot_data:
        plot_data(dataset_train, title='Train set')
        plot_data(dataset_test, title='Test set')
    
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    
    model = Net(for_quantization=True)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    summary(model, (1, 16, 16))
    
    # model must be set to eval for fusion to work
    model.eval()

    # Fuse Conv, and relu
    model = torch.ao.quantization.fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2'], ['conv3', 'relu3'], ['conv4', 'relu4']])
    
    # attach a global qconfig, which contains information about what kind of quantization configuration to use
    model.qconfig = get_quant()
    
    # Prepare the model for quantization. This inserts observers in
    # the model that will observe weight and activation tensors during calibration.
    torch.ao.quantization.prepare(model, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Conv1: After observer insertion \n\n', model.conv1)

    # Calibrate with the training set
    for epoch in range(1, args.epochs + 1):
        test(model, device, train_loader)

    print('Post Training Quantization: Calibration done')
    
    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, fuses modules where appropriate,
    # and replaces key operators with quantized implementations.
    torch.ao.quantization.convert(model, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Conv1: After fusion and quantization, note fused modules: \n\n',model.conv1)

    print("Size of model after quantization")
    #summary(model, (1, 16, 16))
    
    # Caluation of final converted model
    test(model, device, test_loader)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print('Model saved in {}'.format(args.output_path))


if __name__ == '__main__':
    main()
