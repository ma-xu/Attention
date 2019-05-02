import torch
from resnet import *
from se_resnet import *
from p_se_resnet import *
from cp_se_resnet import *
from spp_se_resnet import *
from p_spp_se_resnet import *
from cp_spp_se_resnet import *
from utils.flops_counter import get_model_complexity_info


def test():
    net = cp_spp_se_resnet152()
    y = net((torch.randn(1,3,224,224)))
    print(y.size())

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total params:' +str(pytorch_total_params))
    print('Total params:' + str(pytorch_trainable_params))

    flops, params = get_model_complexity_info(net, (224, 224),as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)


test()