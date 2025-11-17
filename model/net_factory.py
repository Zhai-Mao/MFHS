from model.SAM2UNetL import SAM2UNetL
from model.SAM2UNetS import SAM2UNetS
from model.SAM2UNetT import SAM2UNetT

def net_factory(net_type="unet", in_chns=1, class_num=3, checkpoint_path=None):
    if net_type == "SAM2UNetL":
        net = SAM2UNetL(checkpoint_path=checkpoint_path).cuda()
    elif net_type == "SAM2UNetS":
        net = SAM2UNetS(checkpoint_path=checkpoint_path).cuda()
    elif net_type == "SAM2UNetT":
        net = SAM2UNetT(checkpoint_path=checkpoint_path).cuda()
    else:
        net = None
    return net
