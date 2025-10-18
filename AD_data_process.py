import re
import os
import numpy as np
import pandas as pd
import warnings
import pandas as pd
import os
import tqdm
import numpy as np
import scipy.io as sio
warnings.filterwarnings("ignore")
import pickle
import torch
def get_key(file_name):
    file_name = file_name.split('_')
    key = ''
    for i in range(len(file_name)):
        if file_name[i] == 'rois':
            key = key[:-1]
            break
        else:
            key += file_name[i]
            key += '_'
    return key
def load_data(path1, path2):
    all = {}
    labels = {}
    all_data = []
    label = []
    for filename in path1:
        a = np.loadtxt(filename)
        a = a.transpose()
        all[filename] = a
        all_data.append(a)
        data = pd.read_csv(path2)
        for i in range(len(data)):
            if os.path.basename(get_key(filename)) == data['FILE_ID'][i]:
                if int(data['DX_GROUP'][i]) == 2:
                    labels[filename] = int(data['DX_GROUP'][i]-1)
                    label.append(int(data['DX_GROUP'][i]-1))
                else:
                    labels[filename] = 0
                    label.append(0)
                break
    label = np.array(label)
    return all_data, label
def compute_correlation(matrix):
    num_slices, dim1,_  = matrix.shape
    correlation_matrices = np.zeros((num_slices, dim1, dim1))
    for i in range(num_slices):
        for j in range(dim1):
            correlation_matrices[i] = np.corrcoef(matrix[i][j], rowvar=False)
    return correlation_matrices
#设置滑动的大小和步长
def sliding_window(sample, window_size=100, step=10):
    """
    Apply sliding window on the second dimension of the sample.
    Discard the last part if it's smaller than the window size.
    """
    # if sample.shape[1] < window_size:
    #     window_size=10
    if sample.shape[1] < 100:
        window_size= 10

    windows = []
    for start in range(0, sample.shape[1] - window_size + 1, step):
        end = start + window_size
        windows.append(sample[:, start:end])

    # Handle the remaining part if it's larger than step
    # if sample.shape[1] % step != 0 and sample.shape[1] > len(windows) * step:
    #         last_start = sample.shape[1] - window_size
    #         windows.append(sample[:, last_start:])
    return windows

#超参数需要设置--设置填充到哪个位置
def process_samples(samples,lables):
    """
    Process a list of samples, applying sliding window and generating correlation matrices.
    """
    processed = []
    processed_labels=[]
    for i,sample in enumerate(samples):
        # print(sample.shape)
        windows = sliding_window(sample)
        if windows  is not None:
            correlation_matrices = np.array([np.corrcoef(w) for w in windows])
            # print(correlation_matrices.shape[0])
            # print(correlation_matrices.shape)
            # Pad with zeros if less than 25 slices  #超参数需要设置  28--时序最长为320--存在12个小站点中
            if correlation_matrices.shape[0] < 25:
                padded = np.zeros((25, correlation_matrices.shape[1], correlation_matrices.shape[2]))
                padded[:correlation_matrices.shape[0], :, :] = correlation_matrices
                correlation_matrices = padded
                # print(correlation_matrices.shape)
            # if correlation_matrices.shape[0] > 26:
            #     print('False')

            processed.append(correlation_matrices)
            processed_labels.append(lables[i])

    # Combine all processed samples into a single array
    return np.array(processed), np.array(processed_labels)

def get_filtered_files(directory, prefixes):
    """
    遍历目录并筛选以指定前缀开头的文件。

    :param directory: 要遍历的目录路径
    :param prefixes: 前缀列表（如 ['NYU', 'UCLA']）
    :return: 筛选文件的完整路径列表
    """
    filtered_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否以指定前缀开头
            if any(file.startswith(prefix) for prefix in prefixes):
                filtered_files.append(os.path.join(root, file))
    return filtered_files

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
#找到mask的操作
def find_padded_parts(matrix):
    """
    For a given 4D matrix, determine which slices in the second dimension are padded.
    Return a boolean matrix of shape (140, 25, 25) indicating original (True) or padded (False) parts.
    """
    original = np.any(matrix != 0, axis=(2, 3))  # Check for non-zero values in the last two dimensions
    padded_matrix = np.repeat(original[:, :, np.newaxis], 25, axis=2)
    padded_matrix = torch.from_numpy(padded_matrix)
    return padded_matrix
# 您提供的原始标签数据
label_data = """
I247209 LMCI
I247983 EMCI
I248233 EMCI
I248516 AD
I249144 EMCI
I255135 EMCI
I255318 EMCI
I256635 EMCI
I257106 EMCI
I258448 EMCI
I261166 LMCI
I261319 EMCI
I264416 EMCI
I267490 LMCI
I267918 EMCI
I269694 LMCI
I272407 AD
I273181 LMCI
I277286 LMCI
I278367 LMCI
I279103 EMCI
I279181 EMCI
I280800 LMCI
I281024 LMCI
I283853 LMCI
I285812 LMCI 
I286477 LMCI
I287005 MCI
I287082 LMCI
I290305 EMCI
I291229 AD
I295969 AD
I297616 EMCI
I297847 AD
I298203 LMCI
I300057 LMCI
I300841 EMCI
I301221 EMCI
I303143 LMCI
I303434 LMCI
I305240 EMCI
I306073 AD
I306672 EMCI
I306889 LMCI
I310925 AD
I311357 LMCI
I314327 LMCI 
I316276 EMCI
I317397 EMCI
I318774 LMCI
I320432 EMCI
I322000 AD
I322050 MCI
I322371 AD
I323338 LMCI
I323821 EMCI
I324478 EMCI
I330233 LMCI
I332584 EMCI
I332647 MCI
I335317 LMCI
I339129 LMCI
I339436 EMCI
I341806 LMCI
I341823 LMCI 
I341972 AD
I341976 EMCI
I342476 LMCI
I342514 AD
I343366 LMCI
I343571 AD
I346237 AD
I346564 LMCI
I346592 EMCI
I347402 AD
I350113 EMCI
I352722 EMCI
I352947 MCI
I353265 AD
I354598 EMCI
I354654 MCI
I354814 EMCI
I358811 AD
I358857 AD
I359770 AD
I360317 AD
I361116 EMCI 
I361367 EMCI
I361973 EMCI
I362399 LMCI
I363190 AD
I364453 LMCI
I364929 LMCI
I367161 LMCI
I367820 EMCI
I368413 AD
I368901 LMCI
I369264 AD
I370085 AD
I372254 AD
I372938 EMCI
I373417 AD
I374144 EMCI
I374515 EMCI
I375151 CN
I375331 AD
I375500 LMCI
I376259 AD
I376933 AD
I377061 EMCI
I378012 EMCI
I379705 AD
I381307 AD
I382187 AD
I383934 EMCI
I385034 AD
I391167 CN
I392890 EMCI
I394511 LMCI
I398432 LMCI
I398911 AD
I400431 AD
I402257 EMCI
I404042 EMCI
I416213 EMCI
I420457 EMCI
I422066 MCI
I431494 SMC
I472273 EMCI
I180734 EMCI
I192250 EMCI
I207341 EMCI
I217608 EMCI
I227595 MCI
I235258 EMCI
I242177 EMCI
"""

# 自动生成label_mapping字典
label_mapping = {}
pattern = re.compile(r'I(\d+)\s+([A-Z]+)')

for line in label_data.split('\n'):
    line = line.strip()
    if line:
        # 处理一行中有多个ID-标签对的情况
        parts = line.split()
        for i in range(0, len(parts), 2):
            if i+1 < len(parts):
                id_part = parts[i]
                label = parts[i+1]
                # 提取纯数字ID
                id_num = id_part.lstrip('I')
                label_mapping[id_num] = label

# 打印生成的完整label_mapping
print("自动生成的label_mapping字典:")
print("{")
for id_num, label in sorted(label_mapping.items()):
    print(f"    '{id_num}': '{label}',")
print("}")

# 验证数量
print(f"\n共生成 {len(label_mapping)} 个ID-标签映射")

# 设置ABNI文件夹路径（请根据实际路径修改）
# 设置ABNI文件夹路径（请根据实际路径修改）
abni_folder = r'D:\\Desktop\\ADNI\\HarvardOxford' #(134, 25, 118, 118)
#abni_folder = r'D:\Desktop\ADNI\AAL'  #(134, 25, 116, 116)
# abni_folder = r'D:\Desktop\ADNI\BASC\networks' #transformed_data (134, 25, 122, 122)
# abni_folder = r'D:\Desktop\ADNI\BASC\regions' #transformed_data (134, 25, 169, 169)
# abni_folder = r'D:\Desktop\ADNI\MODL\64' #transformed_data (134, 25, 64, 64)
# abni_folder = r'D:\Desktop\ADNI\MODL\128' #transformed_data (134, 25, 128, 128)
# abni_folder = r'D:\Desktop\ADNI\Power' #(134, 25, 264, 264)


import os
import numpy as np
features_list=[]
labels_list=[]
# 遍历文件夹中的文件
for filename in os.listdir(abni_folder):
    if filename.endswith('.txt') and 'timeseries' in filename.lower():
        print(filename)
        # 去掉"I"和"_timeseries.txt"部分
        match = filename[1:].split('_')[0]
        if match:
            file_id = match

            # 读取文件内容作为特征
            filepath = os.path.join(abni_folder, filename)
            try:
                # 获取对应的标签
                label = label_mapping.get(file_id, 'Unknown')  # 如果找不到则标记为Unknown

                # 如果标签是CN则跳过该样本
                if label == 'CN':
                    print(f"跳过CN样本: {filename} -> ID: {file_id}")
                    continue

                # 假设文件内容是数值数据，每行一个特征
                data = np.loadtxt(filepath)
                data = data.transpose()
                features_list.append(data)
                labels_list.append(label)

                print(f"成功处理: {filename} -> ID: {file_id} -> 标签: {label}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
print('labels_list',labels_list)
labels_list = np.array([1 if dx == 'AD' else 0 for dx in labels_list])
print('labels_list',labels_list)
# 转换为numpy数组
raw_data = features_list
labels = labels_list

import numpy as np
from sklearn.model_selection import KFold
import os

# 1. 准备数据
raw_data = np.array(features_list)  # (136, ROI, 时间点)
labels = np.array(labels_list)  # (136,)

# 2. 按标签分离数据
class0_idx = np.where(labels == 0)[0]  # 前96个样本索引
class1_idx = np.where(labels == 1)[0]  # 后40个样本索引

class0_data = raw_data[class0_idx]  # (96, ROI, 时间点)
class1_data = raw_data[class1_idx]  # (40, ROI, 时间点)

# 3. 创建保存结果的文件夹
# os.makedirs('cross_validation_folds', exist_ok=True)


# 4. 定义计算PCC的函数
def calculate_pcc(data):
    raw_data=data
    # 创建一个全 0 的 labels 列表（适用于二分类或多分类）
    labels = [0] * raw_data.shape[0]  # 长度 = raw_data.shape[0]
    transformed_data,process_lables=process_samples(raw_data,labels)
    #4.去除nan,inf
    transformed_data=np.nan_to_num(transformed_data, nan=0, posinf=1, neginf=1)
    print('transformed_data',transformed_data.shape)
    # print('process_lables',process_lables)
    #5.维度转化
    #将numpy的x转换为torch
    X = torch.from_numpy(transformed_data).float()
    #保存mask
    mask = find_padded_parts(transformed_data)

    #矩阵维度：(392, 86, 200, 200)-->(392, 86, X)
    X1=triu_x(X)
    numpy_array = X1.numpy()
    print('numpy_array',numpy_array.shape)

    return numpy_array,mask


def split_with_overlap(data, n_segments=3):
    """划分时间轴为三段，允许重叠"""
    n_samples, n_roi, n_time = data.shape

    # 检查时间轴长度是否足够
    if n_time < 135:
        raise ValueError(f"Time dimension ({n_time}) must be at least 135 for this split method.")

    # 定义每段的起始和结束索引
    segments_ranges = [
        (1, 45),  # 第一段：1-95
        (45, 90),  # 第二段：20-115
        (90, 135)  # 第三段：40-135
    ]

    segments = []
    for start, end in segments_ranges:
        segments.append(data[:, :, start - 1:end])  # Python 索引从 0 开始，所以 start-1


    segments_pcc=[]
    segments_pcc_mask=[]
    # 计算每个segment的PCC并平均
    for seg in segments:  # 遍历每个segment (32, 116, 15)
        pcc_seg, pcc_seg_mask = calculate_pcc(seg)
        segments_pcc.append(pcc_seg)
        segments_pcc_mask.append(pcc_seg_mask)
    segments_pcc = np.array(segments_pcc)
    # 沿 dim=0 合并（最终形状 [96, 25, 25]）
    merged_tensor = torch.cat(segments_pcc_mask, dim=0)
    # print('merged_tensor',merged_tensor.shape)  # torch.Size([96, 25, 25])
    # # 合并第0维和第1维
    segments_pcc = segments_pcc.reshape(-1,25 ,6670)  # -1表示自动计算维度大小
    print('segments_pcc',segments_pcc.shape)
    segments_pcc_mask = merged_tensor
    print('segments_pcc_mask', segments_pcc_mask.shape)
    return segments_pcc,segments_pcc_mask


# 5. 五折交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

class1_folds = list(kf.split(class1_data))




for fold_idx, (class0_train_idx, class0_test_idx) in enumerate(kf.split(class0_data)):
    # 获取class1对应的划分（保持相同的折数）
    # class1_train_idx, class1_test_idx = next(kf.split(class1_data))
    class1_train_idx, class1_test_idx = class1_folds[fold_idx]
    print('class1_train_idx',class1_train_idx)
    print('class1_test_idx',class1_test_idx)

    train_data_0=class0_data[class0_train_idx]
    train_pcc_0,train_pcc_0_mask = calculate_pcc(train_data_0)
    print('train_pcc_0',train_pcc_0.shape)

    train_data_1=class1_data[class1_train_idx]
    train_pcc_1,train_pcc_1_mask=split_with_overlap(train_data_1)
    print('train_pcc_1', train_pcc_1.shape)

    # 6. 获取划分后的数据
    # 训练集
    train_pcc = np.concatenate([
        train_pcc_0,
        train_pcc_1
    ], axis=0)

    import torch

    # 假设 train_pcc_0_mask 和 train_pcc_1_mask 是 PyTorch 张量
    train_pcc_mask = torch.cat([
        train_pcc_0_mask,
        train_pcc_1_mask
    ], dim=0)  # 沿第0维度（行方向）合并

    print(train_pcc_mask.shape)  # 检查合并后的形状

    cc=train_pcc_1.shape[0]
    train_labels = np.concatenate([
        np.zeros(len(class0_train_idx)),
        np.ones(cc)
    ])

    test_data_0 = class0_data[class0_test_idx]
    test_pcc_0,test_pcc_0_mask = calculate_pcc(test_data_0)
    print('test_pcc_0', test_pcc_0.shape)

    test_data_1 = class1_data[class1_test_idx]
    test_pcc_1,test_pcc_1_mask = split_with_overlap(test_data_1)
    print('test_pcc_1', test_pcc_1.shape)

    import torch

    # 假设 train_pcc_0_mask 和 train_pcc_1_mask 是 PyTorch 张量
    test_pcc_mask = torch.cat([
        test_pcc_0_mask,
        test_pcc_1_mask
    ], dim=0)  # 沿第0维度（行方向）合并

    print('test_pcc_mask',test_pcc_mask.shape)  # 检查合并后的形状

    cc1 = test_pcc_1.shape[0]
    # 测试集
    test_pcc = np.concatenate([
        test_pcc_0,
        test_pcc_1
    ], axis=0)
    test_labels = np.concatenate([
        np.zeros(len(class0_test_idx)),
        np.ones(cc1)
    ])
    # 检查哪些位置的 25 维度是全零（沿 feature_dim 维度）
    is_zero = np.all(test_pcc_0 == 0, axis=2)  # 形状 (x, 25)

    # 打印每个样本的哪些位置是全零
    for sample_idx in range(test_pcc_0.shape[0]):
        zero_positions = np.where(is_zero[sample_idx])[0]
        print(f"样本test_pcc_0 {sample_idx} 的全零位置: {zero_positions}")

    # 检查哪些位置的 25 维度是全零（沿 feature_dim 维度）
    is_zero = np.all(test_pcc_1 == 0, axis=2)  # 形状 (x, 25)

    # 打印每个样本的哪些位置是全零
    for sample_idx in range(test_pcc_1.shape[0]):
        zero_positions = np.where(is_zero[sample_idx])[0]
        print(f"test_pcc_1 {sample_idx} 的全零位置: {zero_positions}")
    # 确保mask数据是numpy格式（如果是torch张量则转换）
    if hasattr(train_pcc_mask, 'numpy'):  # 如果是torch张量
        train_pcc_mask = train_pcc_mask.numpy()
    if hasattr(test_pcc_mask, 'numpy'):  # 如果是torch张量
        test_pcc_mask = test_pcc_mask.numpy()
    # 8. 保存当前折的数据
    fold_dir = f'transformed1234_cross_validation_folds/fold_{fold_idx + 1}'
    os.makedirs(fold_dir, exist_ok=True)
    # 保存训练mask
    path_train_mask = os.path.join(fold_dir, 'train_pcc_mask.npy')
    np.save(path_train_mask, train_pcc_mask)
    # 保存测试mask
    path_test_mask = os.path.join(fold_dir, 'test_pcc_mask.npy')
    np.save(path_test_mask, test_pcc_mask)


    # 假设这些变量已经定义
    # train_pcc, train_labels, test_pcc, test_labels

    # 保存训练数据
    path_train_pcc = os.path.join(fold_dir, 'train_pcc.npy')
    np.save(path_train_pcc, train_pcc)

    # 保存训练标签
    path_train_labels = os.path.join(fold_dir, 'train_labels.npy')
    np.save(path_train_labels, train_labels)

    # 保存测试数据
    path_test_pcc = os.path.join(fold_dir, 'test_pcc.npy')
    np.save(path_test_pcc, test_pcc)

    # 保存测试标签
    path_test_labels = os.path.join(fold_dir, 'test_labels.npy')
    np.save(path_test_labels, test_labels)