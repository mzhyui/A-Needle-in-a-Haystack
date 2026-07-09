import numpy as np
import torch

def CL(net, model_name, base_dir='./logs'):
    import torch.nn as nn
    import matplotlib.pyplot as plt
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            channel_lips = []
            weights_norm = []
            for idx in range(m.weight.shape[0]):
                weight = m.weight[idx]
                weight = weight.reshape(weight.shape[0], -1).cpu()
                channel_lips.append(torch.svd(weight)[1].max())
                weights_norm.append(float(torch.norm(weight)))
            channel_lips = torch.Tensor(channel_lips)

    # print("channel_lips")
    # print(channel_lips)
    plt.figure(figsize=(8, 6))
    plt.hist(channel_lips.numpy(), bins=np.arange(0, channel_lips.max() + 0.02, 0.02), edgecolor='black')
    plt.xlabel('Channel Lipschitz Constant')
    plt.ylabel('Frequency')
    plt.title('Distribution of Channel Lipschitz Constants')
    plt.grid(True)
    plt.savefig(f'{base_dir}/channel_lips_distribution_{model_name}.png')
    plt.close()

