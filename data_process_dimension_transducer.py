import xlwt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import random
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
#将x转换为上三角矩阵--四维矩阵-->三维矩阵
def triu_x(x):
    #确保 x 是上三角矩阵
    x = torch.triu(x)
    #初始化一个空的 tensor 用于存储结果
    result = torch.zeros((x.size(0),x.size(1),(x.size(2)*(x.size(3)-1))//2))
    # 遍历每个矩阵，提取非零元素
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            #获取当前矩阵的非零元素
            non_zero_elements = x[i, j][x[i, j] != 0]
            #获取当前矩阵的不为1的元素
            non_zero_elements = non_zero_elements[non_zero_elements != 1]
            #获取当前矩阵的前6670个元素
            selected_elements = non_zero_elements[:((x.size(2)*(x.size(3)-1))//2)]
            result[i, j, :selected_elements.numel()] = selected_elements
    return result

X = np.load('.\data\\BD_X1.npy')
#将numpy的x转换为torch
X = torch.from_numpy(X).float()
#将x转换为上三角矩阵
# 四维矩阵-->三维矩阵
#矩阵维度：(116, 116)-->(6670)
X1=triu_x(X)

numpy_array = X1.numpy()
print(numpy_array.shape)

# np.save('.\data\\' +  '\\BD_X1_three_dimensions.npy', numpy_array)


