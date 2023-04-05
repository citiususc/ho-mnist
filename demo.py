#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms

from homnist.visualization import DrawingCanvas
from homnist.learning import MinMaxScale, ConvertToBlackWhite
from homnist.network import HONet, HACCNet
        
IMAGE_SIZE = [28, 28]  # MNIST input size
PENCIL_THICKNESS = 2.5
SIZE_MULTIPLIER = 20  # So things are not as small



def main():
    # Demo settings
    parser = argparse.ArgumentParser(description='Hardware Oriented MNIST test')
    parser.add_argument('--model-path', type=str, default="./models/mnist_quantized_converted.pth",
                        help='MNIST trained model after quantization and conversion')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--live', action='store_true', default=False,
                        help='predict while drawing')
    parser.add_argument('--hardware', action='store_true', default=False,
                        help='Tests on real hardware')
    parser.add_argument('--serial-port', type=str, default="/dev/ttyUSB2",
                        help='Serial port for connection to FPGA')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    
    if not args.hardware:
        model = HONet()
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            ConvertToBlackWhite(),
            transforms.Resize([16, 16], interpolation=transforms.InterpolationMode.NEAREST),
            MinMaxScale()
            ])
        
        def process_image(image):
            # cv2.imwrite("my_drawing.png", image)
            with torch.no_grad():
                transformed_image = transform_test(image)
                transformed_image = transformed_image.unsqueeze(0)  # We add the "batch" dimension
                
                results_logits = model(transformed_image)
                results = model.softmax(results_logits)
                
                results = results[0]  # We remove the "batch" dimension
                results = results.detach().cpu().numpy()
                
            return results
    
    else:
        model = HACCNet(weights_dir=args.model_path, port=args.serial_port)
        
        def process_image(image):
            # cv2.imwrite("my_drawing.png", image)
            results = model.forward(image)
            results = model.softmax(results)
                
            return results
    
    canvas = DrawingCanvas(title='MNIST Draw', save_event=process_image, size=IMAGE_SIZE, pencil_thickness=PENCIL_THICKNESS, display_multiplier=SIZE_MULTIPLIER, save_while_drawing=args.live)
    canvas.mainloop()
    
if __name__ == "__main__":
    main()

