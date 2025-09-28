from model.SAM2UNetL import SAM2UNetL

def net_factory(net_type="unet", in_chns=1, class_num=3, checkpoint_path=None):
    if net_type == "SAM2UNetL":
        net = SAM2UNetL(checkpoint_path=checkpoint_path).cuda()
    else:
        net = None
    return net