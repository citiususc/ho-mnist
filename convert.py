#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import torch


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Converts quantized weights to normal (int8) weights')
    parser.add_argument('--model-path', type=str, default="./models/mnist_quantized.pth",
                        help='MNIST quantized qint8 model')        
    parser.add_argument('--output-path', type=str, default="./models/mnist_quantized_converted.pth",
                        help='MNIST model after quantization and conversion')
    args = parser.parse_args()
    
    
    checkpoint_quantized = torch.load(args.model_path)
    
    # HEY! This only works for CNNs without bias
    # HEY! Also, it only workd for quantization with zero_point = 0
    # convX.weight will be the int_repr(convX.weight)
    # convX.kernel_scale will be convX.weight.scale
    # convX.result_scale will be convX.scale
    new_checkpoint_dict = {}
    for parameter_name in list(checkpoint_quantized):
        if parameter_name == 'quant.scale':
            new_parameter_name = 'input_scale'
            quantized_param = checkpoint_quantized[parameter_name]
            assert len(quantized_param.size()) == 1 and quantized_param.size()[0] == 1, "We don't support per-channel quantization yet"
            quantized_param = quantized_param[0]
            new_checkpoint_dict[new_parameter_name] = quantized_param
            
        elif parameter_name.endswith('.weight'):
            parameter_root_name = parameter_name.split('.')[0]
            
            quantized_param = checkpoint_quantized[parameter_name]
            quantized_param_weight = torch.int_repr(quantized_param)
            quantized_param_kernel_scale = torch.Tensor([quantized_param.q_scale()])[0]
            quantized_param_result_scale = checkpoint_quantized[parameter_root_name + '.scale']
            
            new_checkpoint_dict[parameter_name] = quantized_param_weight
            
            parameter_name_k = parameter_root_name + '_kernel_scale'
            new_checkpoint_dict[parameter_name_k] = quantized_param_kernel_scale
            
            parameter_name_r = parameter_root_name + '_result_scale'
            new_checkpoint_dict[parameter_name_r] = quantized_param_result_scale
        
        elif parameter_name.endswith('.scale'):
            # We should already process it in its convolution
            print('I think I\'m processing {}'.format(parameter_name))
        
        else:
            print('Skipping {}'.format(parameter_name))
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(new_checkpoint_dict, args.output_path)
    print('Model saved in {}'.format(args.output_path))

if __name__ == '__main__':
    main()
