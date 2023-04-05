#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import argparse
import torch

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Displays quantized weights after conversion')
    parser.add_argument('--model-path', type=str, default="./models/mnist_quantized_converted.pth",
                        help='MNIST trained model after quantization and conversion')     
    parser.add_argument('--chip-format', action='store_true', default=False,
                        help='Plots weights in chip format')   
    args = parser.parse_args()
    
    checkpoint = torch.load(args.model_path)
    
    print('Found {} parameters: {}'.format(len(checkpoint), list(checkpoint)))
    for p_name in list(checkpoint):
        p_val = checkpoint[p_name].numpy()
        print('============================== {} =============================='.format(p_name))
        
        print('=========== {} | (min={:.2f}, mean={:.2f}, max={:.2f}) ==========='.format(p_val.shape, p_val.min(), p_val.mean(), p_val.max()))
        
        if len(p_val.shape) > 1 and args.chip_format:
            out_chan, in_chan, kernel_height, kernel_width = p_val.shape
            print('// Start of file {}'.format(p_name))
            print('#RC {} {}'.format(in_chan, out_chan))
            
            for o_c in range(out_chan):
                print('// Column number {}'.format(o_c))
                for i_c in range (in_chan):
                    kernel_val = p_val[o_c, i_c]
                    kernel_val = kernel_val.flatten()
                    kernel_val = kernel_val.tolist()
                    kernel_val = ' '.join([str(v) for v in kernel_val])
                    print(kernel_val)
            
            print('// End of file')
        
        else:   
            print(p_val)
        print('==========================================================================')
        
if __name__ == '__main__':
    main()
