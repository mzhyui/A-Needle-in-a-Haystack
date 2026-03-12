import argparse
import time
import numpy as np

import matplotlib.pyplot as plt

from utils.samVisual import fromDict, lossSurface

def donuts(l=-1e1, r=1e1, steps=10):
    # 1. 定义x和y的范围
    x = np.linspace(l, r, steps)  # x轴的范围
    y = np.linspace(l, r, steps)  # y轴的范围
    X, Y = np.meshgrid(x, y)       # 创建网格坐标

    # 2. 定义目标函数 f(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))  # 示例函数，取决于你的问题定义


    # 3. 绘制等高线图
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, levels=5, cmap='viridis')  # 绘制线框等高线图
    # plt.clabel(contour, inline=True, fontsize=8)               # 在等高线上标注数值
    plt.colorbar(contour, label='f(x, y)')                     # 添加颜色条

    plt.title('2D Contour Plot')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(alpha=0.3)
    plt.savefig('./data/2d_contour.png')  # 保存图片

def chessGrid(l=-1e-1, r=1e-1, steps=10):
    x = np.linspace(l, r, steps)  # x轴的范围
    y = np.linspace(l, r, steps)  # y轴的范围
    X, Y = np.meshgrid(x, y)       # 创建网格坐标

    plt.figure(figsize=(8, 6))

    for i, x in enumerate(X):
        for j, y in enumerate(Y):        # 添加颜色条
            plt.scatter(x, y, c='r', s=10, alpha=0.5)

    plt.title('scatter Plot')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(alpha=0.3)
    plt.savefig('./data/scatter.png')  # 保存图片






if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--pre_trained', type=str, default='fl_save/cifar10/resnet20_iidFalse_num1000_C0.1_attacktype-dba/shard5/dba12-09--16-11-30/fed/attack_portion0.2_model_199.pt', help='pretrained model path')
    argparser.add_argument('-x','--pre_trained_x', type=str, default='', help='pretrained x path')
    argparser.add_argument('-y','--pre_trained_y', type=str, default='', help='pretrained y path')
    argparser.add_argument('-l','--l', type=float, default=-1e-1, help='left bound')
    argparser.add_argument('-r','--r', type=float, default=1e-1, help='right bound')
    argparser.add_argument('-p','--parallel', action='store_true', help='parallel')
    argparser.add_argument('-s','--steps', type=int, default=10, help='steps')
    argparser.add_argument('-w','--threads', type=int, default=4, help='threads')
    argparser.add_argument('--level', type=int, default=20, help='levels')
    argparser.add_argument('--from_dict', type=str, default='', help='from_dict')
    args = argparser.parse_args()

    # assert 'cifar10' in args.pre_trained and 'resnet20' in args.pre_trained, 'only support cifar10 and resnet20 now'

    # dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=4)

    donuts(l=args.l, r=args.r, steps=args.steps)

    begin_time = time.time()
    if not args.from_dict:
        lossSurface(pre_trained=args.pre_trained,
                    dataset='cifar10',
                    pre_trained_x=args.pre_trained_x,
                    pre_trained_y=args.pre_trained_y,
                    l=args.l, r=args.r, steps=args.steps,
                    levels=args.level,
                    threads=args.threads,
                    parallel=args.parallel,
                    )
    else:
        fromDict(save_dict_path=args.from_dict,
                 l=args.l, r=args.r, steps=args.steps,
                 levels=args.level,
                )
    elapsed = time.time() - begin_time
    print(f'Elapsed time: {elapsed:.2f}')

    

