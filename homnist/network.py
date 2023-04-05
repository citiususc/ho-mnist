from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary
import numpy as np
from glob import glob

from homnist.learning import MinMaxScale, ConvertToBlackWhite
from homnist import hacc

class HACCNet(object):
    """Hardware Accelerated Net - Runs on a real FPGA and a real 5-bit chip"""
    
    def print_debug(self, *args, **kwargs):
        if self.DEBUG:
            print(*args, **kwargs)
    
    def __init__(self, weights_dir, port='/dev/ttyUSB2', baudrate=1500000, io_clock=3, dtc_clock=8, debug=False):
        super().__init__()
        self.DEBUG = debug
        
        hacc.init_serial(port=port, baudrate=baudrate)
        self.print_debug("Configured serial")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ConvertToBlackWhite(),
            transforms.Resize([16, 16], interpolation=transforms.InterpolationMode.NEAREST),
            MinMaxScale()
            ])
        
        weight_files = glob(os.path.join(weights_dir, "*.wg"))
        weight_files = list(sorted(weight_files))

        rst_return = hacc.reset()
        self.print_debug("Reset ok with return {}".format(rst_return))
        
        hacc.configureioclk(io_clock)
        self.print_debug("Configure IO clock OK")
        
        hacc.configuredtcclk(dtc_clock)  # DTC time multiplier.
        hacc.confaccrelu(int("00010000", 2))  # This word only activates the RELU

        hacc.send_CNN(weight_files)
        self.print_debug("weighs sent")
    
    
    def forward(self, image):  # image is standard MNIST (size=[28,28], min=0, max=255)
        image = self.transform(image)  # torch.Size([1, 16, 16])
        image = image.cpu().detach().numpy()
        
        image = image.transpose(1, 2, 0)  # We move the channel to the last dimension | torch.Size([1, 16, 16])
        image = image * 2.0  # Chip likes [0, 30] range
        
        # compu_img = hacc.image_to_compu(image)
        hacc.send_image_cnn(image)
        hacc.run_cnn()
        out_logits = hacc.read_output_image_cnn()
        out_logits = out_logits[:, 0, 0]  # We remove height, width dimensions
        
        return out_logits
    
    
    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


class HONet(nn.Module):
    """Harware Oriented Net - Tries to emulate an FPGA and 5-bit CNN"""
    
    def print_debug(self, *args, **kwargs):
        if self.DEBUG:
            print(*args, **kwargs)
    
    def debug_tensor_info(self, x, name):
        self.print_debug('  {}:  {} | (min={:.2f}, mean={:.2f}, max={:.2f})'.format(name, x.size(), x.detach().numpy().min(), x.detach().numpy().mean(), x.detach().numpy().max()))
            

    def __init__(self, bias=False, debug=False):
        super().__init__()
        assert bias == False
        self.DEBUG = debug
        
        self.input_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)
        
        self.conv1_kernel_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=bias
        )
        self.relu1 = torch.nn.ReLU()
        self.conv1_result_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)
        
        self.conv2_kernel_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=bias
        )
        self.relu2 = torch.nn.ReLU()
        self.conv2_result_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)
        
        self.conv3_kernel_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias
        )
        self.relu3 = torch.nn.ReLU()
        self.conv3_result_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)
        
        self.conv4_kernel_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)
        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=bias
        )
        self.relu4 = torch.nn.ReLU()
        self.conv4_result_scale = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

        self.softmax = torch.nn.Softmax(dim=-1)
        
        # We will store intermediate results in these variables for debugging
        self.conv1_out = None
        self.conv2_out = None
        self.conv3_out = None
        self.conv4_out = None
    
    def _quantize(self, x, min_val=0, max_val=31):
        # Converts `x` to int (nearest rounding) and saturates to [min_val, max_val]
        x = torch.round(x)
        x = torch.clip(x, min=min_val, max=max_val)
        
        return x
        
        
    def forward(self, x):
        self.print_debug('===================== START FORWARD =====================')
        
        ################################# INPUT #################################
        self.print_debug('=== INPUT ===')
        self.debug_tensor_info(x, 'Input')  # torch.Size([64, 1, 16, 16]) | (min=0.00, mean=1.91, max=15.00)
        
        
        ################################# CONV1 #################################
        self.print_debug('=== CONV1 ===')
        # (In the FPGA)
        x = x / self.input_scale * (self.conv1_kernel_scale / self.conv1_result_scale)  # x = x * 0.18749297  | # HEY! Notice that here we divide by `self.input_scale`
        self.debug_tensor_info(x, 'conv1.input')  # torch.Size([64, 1, 16, 16]) | (min=0.00, mean=0.36, max=2.81)
        
        # (In the CHIP)
        x = self._quantize(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self._quantize(x)
        self.conv1_out = x.cpu().detach().numpy()  # So we can debug intermediate results
        self.debug_tensor_info(x, 'conv1.output')  # torch.Size([64, 16, 14, 14]) | (min=0.00, mean=2.44, max=31.00)
        
        
        ################################# CONV2 #################################
        self.print_debug('=== CONV2 ===')
        # (In the FPGA)
        x = x * self.input_scale * (self.conv2_kernel_scale / self.conv2_result_scale)  # x = x * 0.03320471
        self.debug_tensor_info(x, 'conv2.input')  # torch.Size([64, 16, 14, 14]) | (min=0.00, mean=0.08, max=1.03)
        
        # (In the CHIP)
        x = self._quantize(x)
        x = self.conv2(x)  # Stride=2
        x = self.relu2(x)
        x = self._quantize(x)
        self.conv2_out = x.cpu().detach().numpy()  # So we can debug intermediate results
        self.debug_tensor_info(x, 'conv2.output')  # torch.Size([64, 16, 6, 6]) | (min=0.00, mean=2.87, max=31.00)
        
        
        ################################# CONV3 #################################
        self.print_debug('=== CONV3 ===')
        # (In the FPGA)
        x = x * self.conv1_result_scale * (self.conv3_kernel_scale / self.conv3_result_scale)  # x = x * 0.040664427
        self.debug_tensor_info(x, 'conv3.input')  # torch.Size([64, 16, 6, 6]) | (min=0.00, mean=0.12, max=1.26)
        
        # (In the CHIP)
        x = self._quantize(x)
        x = self.conv3(x)  # Stride=2, Padding=1
        x = self.relu3(x)
        x = self._quantize(x)
        self.conv3_out = x.cpu().detach().numpy()  # So we can debug intermediate results
        self.debug_tensor_info(x, 'conv3.output')  # torch.Size([64, 16, 3, 3]) | (min=0.00, mean=4.31, max=31.00)
        
        
        ################################# CONV4 #################################
        self.print_debug('=== CONV4 ===')
        # (In the FPGA)
        x = x * self.conv2_result_scale * (self.conv4_kernel_scale / self.conv4_result_scale)  # x = x * 0.050527696
        self.debug_tensor_info(x, 'conv4.input')  # torch.Size([64, 16, 3, 3]) | (min=0.00, mean=0.22, max=1.57)
        
        # (In the CHIP)
        x = self._quantize(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self._quantize(x)
        self.conv4_out = x.cpu().detach().numpy()  # So we can debug intermediate results
        self.debug_tensor_info(x, 'conv4.output')  # torch.Size([64, 10, 1, 1]) | (min=0.00, mean=4.69, max=31.00)
        
        
        ################################# OUTPUT ################################
        self.print_debug('=== OUTPUT ==')
        # (In the FPGA)
        x = x * self.conv3_result_scale  # x = x * 0.3558817
        x = x.view(x.size(0), -1)
        self.debug_tensor_info(x, 'Output')  # torch.Size([64, 10]) | (min=0.00, mean=1.67, max=11.03)
        
        # HEY! Remember that we apply the Softmax outside!
        
        self.print_debug('====================== END FORWARD ======================')
        return x


class Net(nn.Module):
    """Net for MNIST classification - Follows some Chip constrains"""
    
    def print_debug(self, text):  # If quantization, variable args are not supported
        if self.DEBUG:
            print(text)
    
    def __init__(self, for_quantization=False, bias=False, debug=False):
        super().__init__()
        self.for_quantization = for_quantization
        self.DEBUG = debug
        
        if self.for_quantization:
            # QuantStub converts tensors from floating point to quantized
            self.quant = torch.quantization.QuantStub()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=bias
        )
        if self.for_quantization:  # If quantization, we explicitly declare the relu to later fuse it. If no quantization, we'll user GeLu/relu for train/test
            self.relu1 = torch.nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=bias
        )
        if self.for_quantization:  # If quantization, we explicitly declare the relu to later fuse it. If no quantization, we'll user GeLu/relu for train/test
            self.relu2 = torch.nn.ReLU()
        
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias
        )
        if self.for_quantization:  # If quantization, we explicitly declare the relu to later fuse it. If no quantization, we'll user GeLu/relu for train/test
            self.relu3 = torch.nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=bias
        )
        if self.for_quantization:  # If quantization, we explicitly declare the relu to later fuse it. If no quantization, we'll user GeLu/relu for train/test
            self.relu4 = torch.nn.ReLU()
        
        # For training/normal inference we will add dropout and noise layers. I'm too scared to add those during quantization
        if not self.for_quantization:
            self.noise = NoiseLayer()
            self.dropout = nn.Dropout(0.5)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
        if self.for_quantization:
            # DeQuantStub converts tensors from quantized to floating point
            self.dequant = torch.quantization.DeQuantStub()
    
    def _activation(self, x):
        assert not self.for_quantization
        # If there are activations on EVERY layer, network doen't learn with relu, as too many neurons die
        if self.training:
            # it's in train mode
            return F.gelu(x)
        
        else:
            # it's in eval mode
            return F.relu(x)
        
    def forward(self, x):
        self.print_debug('===================== START FORWARD =====================')
        
        ################################# INPUT #################################
        self.print_debug('  Input size:  {}'.format(x.size()))  # torch.Size([64, 1, 16, 16])
        if self.for_quantization:
            x = self.quant(x)  # We quantize the input to INT8
        
        if not self.for_quantization:
            x = self.noise(x)  # We add noise to simulate chip noise (or whatever)
        
        ################################# CONV1 #################################
        x = self.conv1(x)
        if self.for_quantization:
            x = self.relu1(x)
        else:
            x = self._activation(x)
        self.print_debug('  After conv1: {}'.format(x.size()))  # torch.Size([64, 16, 14, 14])
        
        if not self.for_quantization:
            x = self.noise(x)  # We add noise to simulate chip noise (or whatever)
        
        ################################# CONV2 #################################
        x = self.conv2(x)
        if self.for_quantization:
            x = self.relu2(x)
        else:
            x = self._activation(x)
        self.print_debug('  After conv2: {}'.format(x.size()))  # torch.Size([64, 16, 6, 6])
        
        if not self.for_quantization:
            x = self.noise(x)  # We add noise to simulate chip noise (or whatever)
        
        ################################# CONV3 #################################
        x = self.conv3(x)
        if self.for_quantization:
            x = self.relu3(x)
        else:
            x = self._activation(x)
        self.print_debug('  After conv3: {}'.format(x.size()))  # torch.Size([64, 16, 3, 3])
        
        if not self.for_quantization:
            x = self.noise(x)  # We add noise to simulate chip noise (or whatever)
            x = self.dropout(x)  # We add dropout
        
        ################################# CONV4 #################################
        x = self.conv4(x)
        if self.for_quantization:
            x = self.relu3(x)
        else:
            x = self._activation(x)
        self.print_debug('  After conv4: {}'.format(x.size()))  # torch.Size([64, 10, 1, 1])
        
        if not self.for_quantization:
            x = self.noise(x)  # We add noise to simulate chip noise (or whatever)
        
        ################################# OUTPUT #################################
        x = x.view(x.size(0), -1)
        self.print_debug('  After view:  {}'.format(x.size()))  # torch.Size([64, 10])
        
        if self.for_quantization:
            x = self.dequant(x)  # We dequantize the input to float
        
        # HEY! Remember that we apply the Softmax outside!
        
        self.print_debug('====================== END FORWARD ======================')
        return x


class NoiseLayer(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = torch.randn(x.size(), requires_grad=False) * scale
            x = x + sampled_noise
        return x

