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
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, src_embed, d_model1, d_model2):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.fc = nn.Sequential(
            nn.Linear(6670, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.fc1=nn.Linear(64, 2)

    def forward(self, src):
        "Take in and process masked src and target sequences."
        src = self.encode(src)
        src = (torch.sum(src, dim=1) / src.size(1)).squeeze()  # torch.Size([16, 256])
        src = self.fc(src)  # 32, 2
        src = self.fc1(src)
        src = F.softmax(src, dim=-1)
        return src

    def encode(self, src):
        return self.encoder(self.src_embed(src))
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        #loss = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return self.norm(x)
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, size2, self_attn1, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn1 = self_attn1
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.sublayer2 = clones(SublayerConnection(size2, dropout), 1)
        self.size = size
    def forward(self, x):
        x = self.sublayer[0](x,lambda x: self.self_attn1(x, x, x))
        return self.sublayer[1](x,self.feed_forward)
def attention(query, key, value):
    "Compute 'Scaled Dot Product Attention'"
    scores = torch.matmul(query, key.transpose(-2, -1))
    p_attn = F.softmax(scores, dim=-2)
    return torch.matmul(p_attn, value)
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        self.h = h
        self.linears = clones(nn.Linear(d_model, 128 * h), 3)
        self.W_o = nn.Linear(128 * h, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.ELU = nn.ELU()
    def forward(self, query, key, value,mask=None):
        if mask is not None:
            #Same mask applied to all h heads.
            mask = mask.unsqueeze(1).expand(-1,self.h, -1,-1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [l(x).view(nbatches, -1, self.h, 128 * self.h).transpose(1, 2) for l, x in
        #                      zip(self.linears, (query, key, value))]
        queries = self.linears[0](query)
        keys = self.linears[1](key)
        values = self.linears[2](value)
        queries = queries.reshape([nbatches, self.h, -1, 128])
        keys = keys.reshape([nbatches, self.h, -1, 128])
        values = values.reshape([nbatches, self.h, -1, 128])
        # 2) Apply attention on all the projected vectors in batch.
        x = attention(queries, keys, values)
        # 3) "Concat" using a view and apply a final linear.
        x = x.contiguous().view(nbatches, -1, self.h * 128)
        return self.W_o(x)
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
def make_model(N=6, d_model1=72, d_model2=116, d_ff=2048, h1=8,dropout=0.5):#N=1, d_model1=512, d_model2=25, d_ff=2048, h1=2, dropout=0.5
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn1 = MultiHeadedAttention(h1, d_model1)#h1==注意力机制的头数   d_model1==transformer-encoder输入中，每个token的维度
    ff = PositionwiseFeedForward(d_model1, d_ff, dropout)#d_model1==transformer-encoder输入中，每个token的维度   d_ff=FFN隐藏层的维度
    position = PositionalEncoding(d_model1, dropout) #位置编码   d_model1==transformer-encoder输入中，每个token的维度
    model = EncoderDecoder(Encoder(EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N),
                           c(position),
                           d_model1, d_model2)#d_model2==凑维度，使得NLP分类维度匹配
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
def find_padded_parts(matrix):
    """
    For a given 4D matrix, determine which slices in the second dimension are padded.
    Return a boolean matrix of shape (140, 25, 25) indicating original (True) or padded (False) parts.
    """
    original = np.any(matrix != 0, axis=(2, 3))  # Check for non-zero values in the last two dimensions
    padded_matrix = np.repeat(original[:, :, np.newaxis], 25, axis=2)
    padded_matrix = torch.from_numpy(padded_matrix)
    return padded_matrix
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
    return {'Total': total_num, 'Trainable': trainable_num}


# 示例用法
# X = np.array(...)  # 用你的特征数据替换这里
# Y = np.array(...)  # 用你的标签数据替换这里

X = np.load('.\data\\MDD_X1_three_dimensions.npy')
Y = np.load('.\data\\\MDD_Y1.npy')


#Y=Y[:123]+[246:]
max_acc = 0
best_epoch=170  # Hyperparameter: needs to be adjusted
seed = 121
setup_seed(seed)
epochs = 200
batch_size = 16
drop_out = 0.5
lr =0.00003#0.00002#0.00015 # 0.0002
decay = 0.01
result = []
epoch=0
acc_list = []
precision_list = []
recall_list = []
f1_list = []
auc_list = []
epoch_list = []

train_index_1=[]
test_index_1=[]
kf = KFold(n_splits=5, random_state=seed, shuffle=True)
kfold_index = 0

for train_index, test_index in kf.split(X):
    kfold_index += 1
    # if kfold_index!=4:
    #     continue
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print('X_train{}'.format(X_train.shape))
    print('X_test{}'.format(X_test.shape), )
    print('Y_train{}'.format(Y_train.shape))
    print('Y_test{}'.format(Y_test.shape))
    # print(test_index)

    # model #N=6, d_model1=72, d_model2=116, d_ff=2048, h1=8, h2=10
    Model = make_model(N=1, d_model1=6670, d_model2=25, d_ff=128, h1=2, dropout=0.5)
    # print(Model)
    Model.to(device)
    # 加载transformer-encoder模型的参数+有效
    # Model.load_state_dict(torch.load('./Model-1/' + str(kfold_index) + '.pt'), strict=False)
    Model.load_state_dict(torch.load('./best-Model/' + str(kfold_index) + '.pt'), strict=False)
    optimizer = torch.optim.Adam(Model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
    loss_fn = nn.CrossEntropyLoss()
    Model.eval()


    test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
    outputs = Model(test_data_batch_dev)

    _, indices = torch.max(outputs, dim=1)
    preds = indices.cpu()

    acc = metrics.accuracy_score(Y_test, preds)
    precision = metrics.precision_score(Y_test, preds)
    reacall = metrics.recall_score(Y_test, preds)
    f1 = metrics.f1_score(Y_test, preds)
    auc = metrics.roc_auc_score(Y_test, preds)

    print('test result', [kfold_index, epoch, acc, precision, reacall, f1, auc])
    result.append([epoch, acc, precision, reacall, f1, auc])

    # print('preds',preds)
    # print('lables',Y_test)

    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(reacall)
    f1_list.append(f1)
    auc_list.append(auc)
    epoch_list.append(epoch)
    # 每一次交叉验证后，保存模型，共保存10次
    # torch.save(Model.state_dict(), './model/' + str(kfold_index) + '.pt')

result_epoch_list = [30, 40, 50, 60, 70, 80, 90, 100]
num = len(result) // 5
for i in range(num):
    if (result[i][0] not in result_epoch_list):
        continue
    acc = (result[i][1] + result[i + num][1] + result[i + 2 * num][1] + result[i + 3 * num][1] + result[i + 4 * num][
        1]) / 5
    precision = (result[i][2] + result[i + num][2] + result[i + 2 * num][2] + result[i + 3 * num][2] +
                 result[i + 4 * num][2]) / 5
    reacall = (result[i][3] + result[i + num][3] + result[i + 2 * num][3] + result[i + 3 * num][3] +
               result[i + 4 * num][3]) / 5
    f1 = (result[i][4] + result[i + num][4] + result[i + 2 * num][4] + result[i + 3 * num][4] + result[i + 4 * num][
        4]) / 5
    auc = (result[i][5] + result[i + num][5] + result[i + 2 * num][5] + result[i + 3 * num][5] + result[i + 4 * num][
        5]) / 5
    print('{}-{}-{}-{}-{:.4}\n'.format(result[i][0], lr, batch_size, decay, acc))

ts_result = [['kfold_index', 'prec', 'recall', 'acc', 'F1', 'auc']]  # 创建一个空列表
for i in range(5):  # 创建一个5行的列表（行）
    ts_result.append([])  # 在空的列表中添加空的列表
num = len(acc_list) // 5
for i in range(num):
    if max_acc < (acc_list[i] + acc_list[i + num] + acc_list[i + 2 * num] + acc_list[i + 3 * num] + acc_list[
        i + 4 * num]) / 5:
        max_acc = (acc_list[i] + acc_list[i + num] + acc_list[i + 2 * num] + acc_list[i + 3 * num] + acc_list[
            i + 4 * num]) / 5
        max_precision = (precision_list[i] + precision_list[i + num] + precision_list[i + 2 * num] + precision_list[
            i + 3 * num] + precision_list[i + 4 * num]) / 5
        max_recall = (recall_list[i] + recall_list[i + num] + recall_list[i + 2 * num] + recall_list[i + 3 * num] +
                      recall_list[i + 4 * num]) / 5
        max_f1 = (f1_list[i] + f1_list[i + num] + f1_list[i + 2 * num] + f1_list[i + 3 * num] + f1_list[
            i + 4 * num]) / 5
        max_auc = (auc_list[i] + auc_list[i + num] + auc_list[i + 2 * num] + auc_list[i + 3 * num] + auc_list[
            i + 4 * num]) / 5
        max_epoch = epoch_list[i]

        ts_result[1] = [1, precision_list[i], recall_list[i], acc_list[i], f1_list[i], auc_list[i]]
        ts_result[2] = [2, precision_list[i + num], recall_list[i + num], acc_list[i + num], f1_list[i + num],
                        auc_list[i + num]]
        ts_result[3] = [3, precision_list[i + 2 * num], recall_list[i + 2 * num], acc_list[i + 2 * num],
                        f1_list[i + 2 * num], auc_list[i + 2 * num]]
        ts_result[4] = [4, precision_list[i + 3 * num], recall_list[i + 3 * num], acc_list[i + 3 * num],
                        f1_list[i + 3 * num], auc_list[i + 3 * num]]
        ts_result[5] = [5, precision_list[i + 4 * num], recall_list[i + 4 * num], acc_list[i + 4 * num],
                        f1_list[i + 4 * num], auc_list[i + 4 * num]]
print('{}-{}-{}-{}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}\n'.format(max_epoch, lr, batch_size, decay, max_acc, max_precision,
                                                           max_recall, max_f1, max_auc))


# 保存结果
def save_xlsx(ts_result):
    runtime_id = 'no_tuning_result-{}'.format(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()))
    f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
    a = np.average(
        [ts_result[1], ts_result[2], ts_result[3], ts_result[4], ts_result[5]], axis=0).tolist()
    a[0] = 'average'
    ts_result.append(a)
    # print(a)
    for j in range(len(ts_result)):
        for i in range(len(ts_result[j])):
            sheet1.write(j, i, ts_result[j][i])  # 写入数据参数对应 行, 列, 值
    f.save(runtime_id + '.xlsx')  # 保存.xls到当前工作目录


save_xlsx(ts_result)