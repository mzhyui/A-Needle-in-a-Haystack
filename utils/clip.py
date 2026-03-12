import copy
import torch

def clipping(w_local: dict, net_local: torch.nn.Module):
    d_w = copy.deepcopy(w_local)
    for k in w_local.keys():
        d_w[k] = w_local[k] - net_local.state_dict()[k]
    d_n = copy.deepcopy(w_local)
    for k in w_local.keys():
        d_n[k] = torch.nn.functional.normalize(
            d_w[k].float(), dim=0)
    for k in w_local.keys():
        w_local[k] = w_local[k] - \
            (torch.nn.functional.normalize(
                d_n[k].float(), dim=0)).long()
    return w_local