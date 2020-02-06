'''
@Time    : 2020/2/1
@Author  : Jingsen Zheng
@File    : learn
@Brief   :
'''

import torch

# a = torch.rand(2, 3, 4)
# print(a)
# b = a.min(-1)
# print(b)
# c = a.min(-1)[0]
# print(c)

# dic = dict(a=1, b=2)
# print(dic['c'])

a = torch.rand(2, 3, 4)
print(a)
print(a[:, [1, 2], []])