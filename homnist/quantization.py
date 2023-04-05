from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver
from torch.quantization.qconfig import QConfig

def get_quant(B=5):
    #B is bits

    ##intB qconfig:
    intB_act_fq = HistogramObserver.with_args(quant_min=0,
                                              quant_max=2**B-1,
                                              dtype=torch.quint8,
                                              qscheme=torch.per_tensor_affine,
                                              reduce_range=False)

    intB_weight_fq = MovingAverageMinMaxObserver.with_args(quant_min=-(2**B)/2+1,
                                                           quant_max=(2**B)/2-1,
                                                           dtype=torch.qint8,
                                                           qscheme=torch.per_tensor_symmetric,
                                                           reduce_range=False)

    intB_qconfig = QConfig(activation=intB_act_fq, weight=intB_weight_fq)

    return intB_qconfig


def load_quantized_checkpoint(model, checkpoint_quantized, checkpoint_base=None):
    state_dict = torch.load(checkpoint_base)
    model.load_state_dict(state_dict)
    
    # model must be set to eval for fusion to work
    model.eval()

    # Fuse Conv, and relu
    model = torch.ao.quantization.fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2'], ['conv3', 'relu3'], ['conv4', 'relu4']])
    
    
    # attach a global qconfig, which contains information about what kind of quantization configuration to use
    model.qconfig = get_quant()
    
    # Prepare the model for quantization. This inserts observers in
    # the model that will observe weight and activation tensors during calibration.
    torch.ao.quantization.prepare(model, inplace=True)
    
    torch.ao.quantization.convert(model, inplace=True)
    state_dict = torch.load(checkpoint_quantized)
    model.load_state_dict(state_dict)

