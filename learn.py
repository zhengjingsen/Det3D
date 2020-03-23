'''
@Time    : 2020/2/1
@Author  : Jingsen Zheng
@File    : learn
@Brief   :
'''

import torch
import numpy as np
import matplotlib.pyplot as plt

# a = torch.rand(2, 3, 4)
# print(a)
# b = a.min(-1)
# print(b)
# c = a.min(-1)[0]
# print(c)

# dic = dict(a=1, b=2)
# print(dic['c'])

# a = np.random.rand(2, 3)
# print(a)
# b = np.stack(a, axis=0)

# cls_scores = []
# cls_scores.append(torch.rand(2, 1, 128, 112))
# cls_scores.append(torch.rand(2, 1, 64, 56))
# cls_scores.append(torch.rand(2, 1, 32, 28))
#
# bbox_preds = []
# bbox_preds.append(torch.rand(2, 7, 128, 112))
# bbox_preds.append(torch.rand(2, 7, 64, 56))
# bbox_preds.append(torch.rand(2, 7, 32, 28))
#
# centerness = []
# centerness.append(torch.rand(2, 1, 128, 112))
# centerness.append(torch.rand(2, 1, 64, 56))
# centerness.append(torch.rand(2, 1, 32, 28))
#
# gt_bboxes = []
# gt_bboxes.append(np.random.rand(14, 7))
# gt_bboxes.append(np.random.rand(15, 7))
#
# gt_labels = []
# gt_labels.append(np.random.rand(14))
# gt_labels.append(np.random.rand(15))
#
# from det3d.models.bbox_heads import fcos_head

a=torch.tensor([[[1, 2, 2], [3, 2, 1]],
                [[2, 1, 2], [1, 3, 2]]],
               dtype=torch.float32)
b=torch.tensor([[2, 2, 2], [3, 3, 3]])
# c = a*b
# print(c)
# d = a-b
# e = b-a
# print(d)
# print(e)

# data = np.loadtxt("/opt/plusai/log/obstacle_assign/time.txt", dtype=np.str)
# print(data.shape)
# time_cost = data[:, 2].astype(np.float32)
#
# plt.figure()
# plt.plot(time_cost)
# plt.xlabel("frame")
# plt.ylabel("cost_time(ms)")
# # plt.savefig("cost_time2.jpg")
#
# print("max: {:.3f}ms, min: {:.3f}ms, avg: {:.3f}ms, <10ms: {:.3f}%".format(
#   np.max(time_cost), np.min(time_cost), np.average(time_cost),
#   np.sum(time_cost<10.) / data.shape[0] * 100))

assert False