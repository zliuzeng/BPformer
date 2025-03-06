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

        self.W_A_1 = nn.Parameter(torch.empty(6670, 3))


        self.W_B_1 = nn.Parameter(torch.empty(3, 128 * 2))


        self.W_C_1 = nn.Parameter(torch.empty(6670, 3))


        self.W_D_1 = nn.Parameter(torch.empty(3, 128 * 2))

        self.W_E_1 = nn.Parameter(torch.empty(6670, 3))
        self.W_F_1 = nn.Parameter(torch.empty(3, 128 * 2))


        self.W_A_4 = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_B_4 = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_A_4, a=math.sqrt(5))
        nn.init.zeros_(self.W_B_4)

        self.W_C_4 = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_D_4 = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_C_4, a=math.sqrt(5))
        nn.init.zeros_(self.W_D_4)

        self.W_E_4 = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_F_4 = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_E_4, a=math.sqrt(5))
        nn.init.zeros_(self.W_F_4)

        self.w = nn.Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        torch.nn.init.uniform_(self.w, a=0, b=1)

        self.bn2 = nn.BatchNorm1d(116, affine=True)

        self.w1 = nn.Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        torch.nn.init.uniform_(self.w1, a=0, b=1)


    def forward(self, src,prompt1):

        w=10*(0.4*self.w+0.6*self.w1)
        w = F.relu(self.bn2(w))
        w = (w + w.T) / 2
        #print(w)
        l1 = torch.norm(w, p=1, dim=1).mean()
        result_matrix=self.w_upper_triangle_values(w)

        src = result_matrix * src


        w1 = prompt1(src, self.W_A_4, self.W_A_1, self.W_C_4, self.W_C_1,
                    self.W_E_4, self.W_E_1)
        w2 = prompt1(src, self.W_B_4, self.W_B_1, self.W_D_4, self.W_D_1,
                    self.W_F_4, self.W_F_1)

        src = self.encode(src, w1[0], w2[0], w1[1], w2[1], w1[2], w2[2])
        src = (torch.sum(src, dim=1) / src.size(1)).squeeze()
        src = self.fc(src)
        src = self.fc1(src)
        src = F.softmax(src, dim=-1)
        return src,0.3*l1

    def encode(self, src,W_A,W_B,W_C,W_D,W_E,W_F):
        return self.encoder(self.src_embed(src),W_A,W_B,W_C,W_D,W_E,W_F)

    def w_upper_triangle_values(self,w):
        # 获取上三角部分的索引
        row_indices, col_indices = torch.triu_indices(w.size(0), w.size(1), offset=1)
        # 提取上三角部分的值
        upper_triangle_values = w[row_indices, col_indices]
        # 将上三角部分的值转换为一个向量
        vector = upper_triangle_values.view(-1)
        # 将向量转换为大小为 [1, 6670] 的矩阵
        result_matrix = vector.view(1, -1)
        return result_matrix
    #def w_attention(self,w):



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x,W_A,W_B,W_C,W_D,W_E,W_F):
        "Pass the input (and mask) through each layer in turn."
        #loss = 0
        for i, layer in enumerate(self.layers):
            x = layer(x,W_A,W_B,W_C,W_D,W_E,W_F)
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
    def forward(self, x,W_A,W_B,W_C,W_D,W_E,W_F):
        x = self.sublayer[0](x,lambda x: self.self_attn1(x, x, x,W_A,W_B,W_C,W_D,W_E,W_F))
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
    def forward(self, query, key, value, W_A,W_B,W_C,W_D,W_E,W_F):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [l(x).view(nbatches, -1, self.h, 128 * self.h).transpose(1, 2) for l, x in
        #                      zip(self.linears, (query, key, value))]
        # print(query.shape)
        # print(W_A.shape)
        # print(W_B.shape)
        queries = self.linears[0](query) + query @ (W_A @ W_B)
        keys = self.linears[1](key) + key @ (W_C @ W_D)
        values = self.linears[2](value) + value @ (W_E @ W_F)
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


class Prompt1(nn.Module):

    def __init__(self,
                  model_dim=6670,temperature=8):
        super().__init__()
        self.model_dim = model_dim
        self.attn_W_down = nn.Linear(self.model_dim, 100, bias=False)
        self.attn_W_up = nn.Linear(100, self.model_dim, bias=False)
        self.attn_non_linear = nn.SiLU()
        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.temperature = temperature

        self.attn_W_down1 = nn.Linear(6670, 100, bias=False)
        self.attn_W_up1 = nn.Linear(100, 256, bias=False)
        self.attn_non_linear1 = nn.SiLU()
        self.layer_norm1 = nn.LayerNorm(256)

    def forward(self,inputs_embeds,W_A, W_A_1,W_C, W_C_1,W_E,W_E_1):# 1 ，3 ， 5 ：目标域  2，4，6：源域
        if W_A.shape[0]==6670:
            q_mul_prefix_emb_added = W_A_1 #torch.cat((W_A_1, W_A), dim=1)  # torch.Size([6670, 12])
            k_mul_prefix_emb_added = W_C_1 #torch.cat((W_C_1, W_C), dim=1)  # torch.Size([6670, 12])
            v_mul_prefix_emb_added = W_E_1 # torch.cat((W_E_1, W_E), dim=1)  # torch.Size([6670, 12])
            mul_prefix_emb_added= torch.stack((q_mul_prefix_emb_added,k_mul_prefix_emb_added,v_mul_prefix_emb_added))#torch.Size([3, 6670, 12])

            x = self.attn_W_down(inputs_embeds)
            x = self.attn_non_linear(x)
            x = self.attn_W_up(x)
            x = self.layer_norm(x)  # torch.Size([32, 25, 6670])
            x, _ = torch.max(x, 1)  # torch.Size([32, 6670])

            attn_scores = (x @ mul_prefix_emb_added) / self.temperature#torch.Size([3, 32, 12])
            # 调整形状,进行求和
            summed_matrix = attn_scores.sum(dim=2)#torch.Size([3, 16, 4])
            # softmax操作
            softmax_matrix = F.softmax(summed_matrix, dim=0)  # torch.Size([16,4])

            w=[]
            for i in range(3):
                if i == 0:
                   W1 = softmax_matrix[i,:].unsqueeze(-1).unsqueeze(-1) * W_A_1
                   W4 = W_A
                if i == 1:
                   W1 = softmax_matrix[i, :].unsqueeze(-1).unsqueeze(-1)* W_C_1
                   W4 = W_C
                if i == 2:
                   W1 = softmax_matrix[i, :].unsqueeze(-1).unsqueeze(-1)  * W_E_1
                   W4 = W_E
                W =W1+W4
                w.append(W)

        else:
            q_mul_prefix_emb_added = W_A_1.transpose(0, 1) #torch.cat((W_A_1, W_A), dim=1)  # torch.Size([6670, 12])
            k_mul_prefix_emb_added = W_C_1.transpose(0, 1) #torch.cat((W_C_1, W_C), dim=1)  # torch.Size([6670, 12])
            v_mul_prefix_emb_added = W_E_1.transpose(0, 1) # torch.cat((W_E_1, W_E), dim=1)  # torch.Size([6670, 12])

            # q_mul_prefix_emb_added = torch.cat((W_A_1, W_A), dim=0).transpose(0, 1)  # torch.Size([256,12])
            # k_mul_prefix_emb_added = torch.cat((W_C_1, W_C), dim=0).transpose(0, 1)  # torch.Size([256,12])
            # v_mul_prefix_emb_added = torch.cat((W_E_1, W_E), dim=0).transpose(0, 1)  # torch.Size([256,12])
            mul_prefix_emb_added = torch.stack(
                (q_mul_prefix_emb_added, k_mul_prefix_emb_added, v_mul_prefix_emb_added))  # torch.Size([3, 256, 12])


            x = self.attn_W_down1(inputs_embeds)
            x = self.attn_non_linear1(x)
            x = self.attn_W_up1(x)
            x = self.layer_norm1(x)  # torch.Size([32, 25, 256])
            x, _ = torch.max(x, 1)  # torch.Size([32, 256])
            #print(x.shape)

            attn_scores = (x @ mul_prefix_emb_added) / self.temperature#torch.Size([3, 32, 12])
            # 调整形状,进行求和
            summed_matrix = attn_scores.sum(dim=2)#torch.Size([3, 16, 4])
            # softmax操作
            softmax_matrix = F.softmax(summed_matrix, dim=0)  # torch.Size([16,4])

            w=[]
            for i in range(3):
                if i == 0:
                    W1 = softmax_matrix[i, :].unsqueeze(-1).unsqueeze(-1) * W_A_1
                    W4 = W_A
                if i == 1:
                    W1 = softmax_matrix[i, :].unsqueeze(-1).unsqueeze(-1) * W_C_1
                    W4 = W_C
                if i == 2:
                    W1 = softmax_matrix[i, :].unsqueeze(-1).unsqueeze(-1) * W_E_1
                    W4 = W_E
                W = W1 +W4
                w.append(W)  # torch.Size([16, 3, 256])
        # torch.Size([16, 3, 256])
        # torch.Size([16, 6670, 3])
        # print(softmax_matrix)
        return w

def make_model(N=6, d_model1=72, d_model2=116, d_ff=2048, h1=8,dropout=0.5):#N=1, d_model1=512, d_model2=25, d_ff=2048, h1=2, dropout=0.5
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    prompt1 = Prompt1(6670)
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
    return model,prompt1

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
#计算正则化prompt约束
def constraint(device,prompt):
    #检查变量 prompt 是否是一个列表（list）类型的对象
    if isinstance(prompt,list):
        sum=0
        for p in prompt:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(prompt)
    else:
        return torch.norm(torch.mm(prompt,prompt.T)-torch.eye(prompt.shape[0]).to(device))
def get_parameter_number(model,prompt):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_num2=sum(p.numel() for p in prompt.parameters())
    print({'Total': total_num, 'Trainable': (trainable_num1+trainable_num2)})
    return {'Total': total_num, 'Trainable': (trainable_num1+trainable_num2)}

# 示例用法
# X = np.array(...)  # 用你的特征数据替换这里
# Y = np.array(...)  # 用你的标签数据替换这里

X = np.load('./data//MDD_X1_three_dimensions.npy')
Y = np.load('./data//MDD_Y1.npy')


best_epoch=190  # Hyperparameter: needs to be adjusted

seed = 122
setup_seed(seed)

epochs = 50
batch_size = 16
drop_out = 0.5


loss_all=[]
result = []
loss_all_5=[]
acc_list = []
precision_list = []
recall_list = []
f1_list = []
auc_list = []
epoch_list = []

max_acc = 0
max_precision = 0
max_recall = 0
max_f1 = 0
max_auc = 0
max_epoch = 0
training_time_result=[]
train_index_1=[]
test_index_1=[]
kf = KFold(n_splits=5, random_state=seed, shuffle=True)
kfold_index = 0
for train_index, test_index in kf.split(X):
    kfold_index += 1
    # if kfold_index!=5:
    #     continue
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print('X_train{}'.format(X_train.shape))
    print('X_test{}'.format(X_test.shape), )
    print('Y_train{}'.format(Y_train.shape))
    print('Y_test{}'.format(Y_test.shape))

    # model #N=6, d_model1=72, d_model2=116, d_ff=2048, h1=8, h2=10
    Model,prompt1 = make_model(N=1, d_model1=6670, d_model2=25, d_ff=128, h1=2, dropout=0.5)
    # print(Model)
    Model.load_state_dict(torch.load('./best-Model/' + str(kfold_index) + '.pt'), strict=False)

    # W_A_1
    Model.W_A_1.data=torch.load('./Model-1/' + str(kfold_index) + '.pt')['W_A']
    # W_B_1
    Model.W_B_1.data = torch.load('./Model-1/' + str(kfold_index) + '.pt')['W_B']

    # W_C_1
    Model.W_C_1.data = torch.load('./Model-1/' + str(kfold_index) + '.pt')['W_C']

    # W_D_1
    Model.W_D_1.data = torch.load('./Model-1/' + str(kfold_index) + '.pt')['W_D']


    # W_E_1
    Model.W_E_1.data = torch.load('./Model-1/' + str(kfold_index) + '.pt')['W_E']

    # W_F_1
    Model.W_F_1.data = torch.load('./Model-1/' + str(kfold_index) + '.pt')['W_F']

    #w
    Model.w.data = torch.load('./Model-2/' + str(kfold_index) + '.pt')['w']




    Model.to(device)
    prompt1.to(device)
    for p in  Model.parameters():
            p.requires_grad = False
    for name, param in Model.named_parameters():
        # if "promptc" in name:
        #     param.requires_grad = True
        if "W_A_4" in name:
            param.requires_grad = True
        if "W_B_4" in name:
            param.requires_grad = True
        if "W_C_4" in name:
            param.requires_grad = True
        if "W_D_4" in name:
            param.requires_grad = True
        if "W_E_4" in name:
            param.requires_grad = True
        if "W_F_4" in name:
            param.requires_grad = True
        if "w1" in name:
            param.requires_grad = True
    total_num, trainable_num = get_parameter_number(Model,prompt1)

    # 定义学习率
    prompt_lr = 0.00005#5
    model_lr = 0.00008

    # 创建参数组列表
    model_param_group = []
    # 将 prompt_model 的参数添加到参数组列表中，并设置对应的学习率
    model_param_group.append({"params": prompt1.parameters(),"lr": prompt_lr})

    # 将 Model 的参数添加到参数组列表中，并设置对应的学习率
    #model_param_group.append({"params": Model.parameters(),"lr": model_lr})
    model_param_group.append({"params": Model.W_A_4, "lr": model_lr})
    model_param_group.append({"params": Model.W_B_4, "lr": model_lr})
    model_param_group.append({"params": Model.W_C_4, "lr": model_lr})
    model_param_group.append({"params": Model.W_D_4, "lr": model_lr})
    model_param_group.append({"params": Model.W_E_4, "lr": model_lr})
    model_param_group.append({"params": Model.W_F_4, "lr": model_lr})
    model_param_group.append({"params": Model.w, "lr": lr})


    # 创建优化器
    optimizer = torch.optim.Adam(model_param_group, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)

    loss_fn = nn.CrossEntropyLoss()
    import time
    start_time = time.time()
    # train
    for epoch in range(0, epochs + 1):
        Model.train()
        idx_batchs = np.random.permutation(int(X_train.shape[0]))
        for i in range(0, int(X_train.shape[0]) // int(batch_size)):
            idx_batch = idx_batchs[i * int(batch_size):min((i + 1) * int(batch_size), X_train.shape[0])]

            train_data_batch = X_train[idx_batch]

            train_label_batch = Y_train[idx_batch]

            train_data_batch = torch.from_numpy(train_data_batch).float().to(device)

            train_label_batch = torch.from_numpy(train_label_batch).long()

            optimizer.zero_grad()
            outputs,loss_cc = Model(train_data_batch,prompt1=prompt1)
            outputs = outputs.cpu()
            #print(outputs)
            loss = F.cross_entropy(outputs, train_label_batch, reduction='mean')+loss_cc
            #loss=loss + 0.01 * constraint(device, Model.get_mul_prompt()),weight=torch.tensor([0.85,1])
            loss.backward()
            optimizer.step()

        count = 0
        acc = 0
        if epoch % 5 == 0:
            count = 0
            acc = 0
            for i in range(0, int(X_train.shape[0]) // int(batch_size)):
                idx_batch = idx_batchs[i * int(batch_size):min((i + 1) * int(batch_size), X_train.shape[0])]
                train_data_batch = X_train[idx_batch]
                train_label_batch = Y_train[idx_batch]


                train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch = torch.from_numpy(train_label_batch).long()

                outputs,_ = Model(train_data_batch,prompt1=prompt1)
                _, indices = torch.max(outputs, dim=1)
                preds = indices.cpu()
                acc += metrics.accuracy_score(preds, train_label_batch)
                count = count + 1
            # 训练的loss输出可能有误
            print('train\tepoch: %d\tloss: %.4f\t\tacc: %.4f' % (epoch, loss.item(), acc / count))
            loss_all.append(loss.data.item())

        if epoch % 5 == 0:
            Model.eval()

            test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
            outputs,_  = Model(test_data_batch_dev,prompt1=prompt1)

            _, indices = torch.max(outputs, dim=1)
            preds = indices.cpu()

            acc = metrics.accuracy_score(Y_test, preds)
            precision = metrics.precision_score(Y_test, preds)
            reacall = metrics.recall_score(Y_test, preds)
            f1 = metrics.f1_score(Y_test, preds)
            auc = metrics.roc_auc_score(Y_test, preds)
            # print('preds',preds)
            # print('lables',Y_test)
            print('test result', [kfold_index, epoch, acc, precision, reacall, f1, auc])
            result.append([epoch, acc, precision, reacall, f1, auc])

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(reacall)
            f1_list.append(f1)
            auc_list.append(auc)
            epoch_list.append(epoch)
        if epoch==best_epoch:
            torch.save(Model.state_dict(), './M-Model/' + str(kfold_index) + '.pt')
            torch.save(prompt1.state_dict(), './M-prompt1/' + str(kfold_index) + '.pt')
        # if epoch==50:
        #     break
     # 计算训练时间
    end_time = time.time()
    training_time = round((end_time - start_time), 2)
    print(f"Training time: {training_time:.2f} seconds")
    training_time_result.append(training_time)
    loss_all_5.append(loss_all)

from openpyxl import Workbook
sums = [sum(sublist)/ len(sublist) for sublist in zip(*loss_all_5)]
wb = Workbook()
ws = wb.active
ws.append(sums)
wb.save("example_prompt.xlsx")
print(training_time_result)
for index,i in enumerate(training_time_result):
    print('第%d次循环所用的时间是%s'%(index+1,round(i,4)))
print('总共5次循环所用的时间是%s'%(round(sum(training_time_result)/len(training_time_result),4)))
result_epoch_list = [30, 40, 50, 60, 70, 80, 90, 100]
num = len(result) // 5
for i in range(num):
    if (result[i][0] not in result_epoch_list):
            continue
    acc = (result[i][1] + result[i + num][1] + result[i + 2 * num][1] + result[i + 3 * num][1] +
               result[i + 4 * num][
                   1]) / 5
    precision = (result[i][2] + result[i + num][2] + result[i + 2 * num][2] + result[i + 3 * num][2] +
                     result[i + 4 * num][2]) / 5
    reacall = (result[i][3] + result[i + num][3] + result[i + 2 * num][3] + result[i + 3 * num][3] +
                   result[i + 4 * num][3]) / 5
    f1 = (result[i][4] + result[i + num][4] + result[i + 2 * num][4] + result[i + 3 * num][4] + result[i + 4 * num][
            4]) / 5
    auc = (result[i][5] + result[i + num][5] + result[i + 2 * num][5] + result[i + 3 * num][5] +
               result[i + 4 * num][
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
        runtime_id = 'target_MDD_result-{}'.format(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()))
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
