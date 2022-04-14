import time
 
import torch
from torch import nn
 
 
class MLP(nn.Module):
    '''widths [in_channel, ..., out_channel], with ReLU within'''
 
    def __init__(self, widths, bn=True, p=0.5):
        super(MLP, self).__init__()
        self.widths = widths
        self.bn = bn
        self.p = p
        self.layers = []
        for n in range(len(self.widths) - 2):
            layer_ = nn.Sequential(nn.Linear(self.widths[n], self.widths[n + 1]).cuda(), nn.Dropout(p=self.p).cuda(), nn.ReLU6(inplace=True).cuda(), )  #线性层模块
            self.layers.append(layer_)
        self.layers.append(nn.Sequential(nn.Linear(self.widths[-2], self.widths[-1]), nn.Dropout(p=self.p)))    #最后加一个回归
        self.mlp = nn.Sequential(*self.layers).cuda()
        if self.bn: 
            self.mlp = nn.Sequential(*self.layers, nn.BatchNorm1d(self.widths[-1]).cuda())
 
    def forward(self, x):
 
        for i, layer in enumerate(self.mlp):
            x=layer(x)
        return x
        # return self.mlp(x)
 
if __name__ == '__main__':
    widths=[192,96,84]
 
    model=MLP(widths)
 
    size=224
    input = torch.rand(2, 192).cuda()
    for i in range(2):
        t1 = time.time()
        loc = model(input)
 
        cnt = time.time() - t1
        print(cnt, loc.size())
 