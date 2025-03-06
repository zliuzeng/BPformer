import numpy as np
#对时间序列进行滑动窗口的划分
def sliding_window(sample, window_size=100, step=10):
    """
    Apply sliding window on the second dimension of the sample.
    Discard the last part if it's smaller than the window size.
    """
    if sample.shape[1] < window_size:
        return None  # Discard the sample if it's too small for even one window

    windows = []
    for start in range(0, sample.shape[1] - window_size + 1, step):
        end = start + window_size
        windows.append(sample[:, start:end])

    return windows
#处理每一个样本，计算时间序列的相似度
def process_samples(samples,lables):
    """
    Process a list of samples, applying sliding window and generating correlation matrices.
    """
    processed = []
    processed_labels=[]
    for i,sample in enumerate(samples):
        # print(sample.shape)#(116, 950)
        windows = sliding_window(sample)
        if windows is not None:
            correlation_matrices = np.array([np.corrcoef(w) for w in windows])

            processed.append(correlation_matrices)
            processed_labels.append(lables[i])
    # Combine all processed samples into a single array
    return np.array(processed), np.array(processed_labels)

print('loading  data...')
X = np.load('.\data\\BP_HC_sig.npy')
Y = np.load('.\data\\BP_HC_label.npy')
c=X.transpose(0,2,1)


#1.划分时间窗+计算相关性pcc
transformed_data,process_lables=process_samples(c,Y)

# print(transformed_data.shape)#(X,X ,116, 116)
# print(process_lables.shape)#(X,)
#2.去除nan,inf
transformed_data=np.nan_to_num(transformed_data, nan=0, posinf=1, neginf=1)
#3.保存
# np.save('./data//' +   '//BD_Y1.npy', process_lables)
# np.save('./data//' +  '//BD_X1.npy', transformed_data)

