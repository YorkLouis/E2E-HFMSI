
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats
from torch.utils.data import Dataset  # Dataset是抽象类，不能被实例化，只能继承
from torch.utils.data import DataLoader  # 帮助我们加载数据
import os
import pickle
import datetime
import empyrical
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import optuna  

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
path = os.path.abspath(r'/root/Data-MSI-30')
path1 = os.path.abspath(r'/root/Result-MSI-30/E2E-HFMSI')


####定义所需函数####
def Sampleset(data, win):
    if data.ndim == 3:
        sample = torch.zeros(data.size(0), win, data.size(1), data.size(2))
    else:
        sample = torch.zeros(data.size(0) - win, win, data.size(1))
    for t in range(data.size(0) - win):
        sample[t,] = data[t:t + win, ]
    return sample


class MydataM(Dataset):
    def __init__(self, Input, Lables):
        # 划分数据和标签
        self.ys_data = Lables
        self.x_data = Input
        self.length = len(self.ys_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x_data[index], self.ys_data[index]


def Object(predicted_portfolio, monthly_rr_batch, tc):
    # predicted_portfolio: [N, Asset, T]
    # monthly_rr_batch: [N, Asset, T] (这里假设是价格比 1+r，即 Gross Return)
    
    # --- 1. 计算每一期的组合总收益 (Gross Portfolio Return) ---
    # 维度变化: [N, Asset, T] -> [N, T]
    gross_return = (predicted_portfolio * monthly_rr_batch).sum(dim=1)
    
    # --- 2. 计算扣费后的净收益 (Net Return) ---
    if tc > 0:
        # 计算相邻时间步的权重变化: |w_t - w_{t-1}|
        # 结果维度: [N, T-1]
        # 对应的是从 t=1 到 t=end 的时间段
        w_t = predicted_portfolio[:, :, 1:]
        w_t_prev = predicted_portfolio[:, :, :-1]
        
        # 在资产维度求和，得到每个时间步的总换手率
        turnover = torch.abs(w_t - w_t_prev).sum(dim=1)
        
        # 计算该时间步的交易成本
        # 0.5 是因为买卖各算一次换手，但费率通常针对单边或总额，这里保留您原有的 0.5 逻辑
        cost = turnover * tc * 0.5
        
        # 【关键步骤】维度对齐
        # 因为换手率少了一个时间步(T-1)，我们需要把收益率也切片成对应的(T-1)
        # 逻辑：w_{t-1} 调整到 w_t 产生的成本，应该从 w_t 产生的收益 r_t 中扣除
        gross_return_aligned = gross_return[:, 1:]
        
        # 计算净收益：(1 + r) - cost
        # 注意：这里假设 monthly_rr_batch 是 (1+r)。如果是纯 r，这里逻辑一样。
        net_return = gross_return_aligned*(1 - cost)
        
    else:
        # 如果没有交易成本，直接使用全部收益
        net_return = gross_return

    # --- 3. 计算对数累积收益 (Log Cumulative Return) ---
    # 加入 clamp 防止净收益 <= 0 导致 log 报错 (NaN)
    # 1e-8 是一个极小值，防止 log(0)
    net_return_safe = torch.clamp(net_return, min=1e-8)
    
    # 先取对数，再在时间维度求和
    log_return = torch.log(net_return_safe).sum(dim=1)
    
    # 对 Batch 求平均作为最终 Loss
    ret = log_return.mean()
    
    return ret


def PortfolioGe(score_s):
    softmax = torch.nn.Softmax(dim=0)
    bc = torch.zeros_like(score_s)
    for s in range(score_s.size(0)):  # torch.exp(score[s][t,]) / (torch.exp(score[s][t,])).sum()
        # print(softmax(score_s[s,:,:]).size(),bc[s,:].size())
        bc[s, :] = softmax(score_s[s, :, :])  # [A,L]---[L,1]
    # print(bc.sum(1))
    return bc


def Return_test(bc, rp, tc):  # [T,A]
    # --- 修改：确保设备一致性 ---
    # 1. 确保真实收益率 rp 与 预测权重 bc 在同一个设备上
    if rp.device != bc.device:
        rp = rp.to(bc.device)
        
    # 计算日收益
    bc_1 = torch.zeros_like(bc)
    for t in range(0, bc.size(0) - 1):
        bc_1[t,] = bc[t + 1,]
    diff = (bc_1 - bc).abs().sum(1)
    diff[-1,] = 0
    
    # 2. 创建 torch.ones 时指定设备，使其与 bc 保持一致
    ones = torch.ones(1, bc.size(0)).to(bc.device)
    
    # 计算
    # 注意：这里 tc * diff 也是在 bc.device 上
    r = (bc * rp).sum(1) * (ones - tc/2 * diff)
    return r


###对照策略##
def Market(rp_testset):
    tensor = torch.ones((2,), dtype=torch.float16)
    # bc_m = tensor.new_full((1, rp_testset.size(1)), 1 / rp_testset.size(1))  # 第一期均匀投资，之后一直持有到结束
    cr_im = torch.zeros(rp_testset.size(0), rp_testset.size(1))
    cr_m = torch.zeros(rp_testset.size(0))
    daily_ret = torch.zeros(rp_testset.size(0))
    for t in range(rp_testset.size(0)):
        if t == 0:
            cr_im[t, :] = rp_testset[t, :] / rp_testset.size(1)
            cr_m[t] = cr_im[t, :].sum()
            daily_ret[t] = cr_m[t] / 1
        else:
            cr_im[t, :] = rp_testset[t, :] * cr_im[t - 1, :]
            cr_m[t] = cr_im[t, :].sum()
            daily_ret[t] = cr_m[t] / cr_m[t-1]
    # print(daily_ret)
    return daily_ret, cr_m


def Best(rp_testset):
    cr_im = torch.zeros(rp_testset.size(0), rp_testset.size(1))
    cr_m = torch.zeros(rp_testset.size(0))
    best_p = torch.zeros(1, rp_testset.size(1))
    top_p = torch.zeros_like(best_p)
    bott_p = torch.zeros_like(best_p)
    for t in range(rp_testset.size(0)):
        if t == 0:
            cr_im[t, :] = rp_testset[t, :] / rp_testset.size(1)
            cr_m[t] = cr_im[t, :].sum()
        else:
            cr_im[t, :] = rp_testset[t, :] * cr_im[t - 1, :]
            cr_m[t] = cr_im[t, :].sum()
    finret = cr_im[-1:, ]
    max_valu, max_ind = finret.max(1)
    bott_valu, bott_ind = finret.min(1)
    best_p[0, max_ind] = 1
    bott_p[0, bott_ind] = 1
    best_m = rp_testset.size(1) * cr_im[:, max_ind]
    return rp_testset[:,max_ind], best_m


class Strategy:
    # def __init__(self, df_rel_price: pd.DataFrame):
    #     self.df_rel_price = df_rel_price
    def summary(self):
        pass

    def run(self, df: torch.Tensor, *args, **kwargs):
        pass


class BCRP(Strategy):
    def __init__(self):
        pass

    def run(input):
        # --- 修改开始：数据类型转换 ---
        # 1. 确保输入转为 CPU 端的 numpy 数组
        if torch.is_tensor(input):
            input_np = input.cpu().detach().numpy()
        else:
            input_np = np.array(input)
        
        # 2. 强制转换为 float64 (8字节)，满足 scipy 的要求
        input_np = input_np.astype(np.float64)
        # --- 修改结束 ---

        # Set up constrained optimization
        # 注意：这里使用 input_np 替代了原来的 input
        func = lambda w: -np.prod(np.matmul(input_np, w))
        
        cons = [{'type': 'eq',
                 'fun': lambda w: np.array([sum(w) - 1])},
                {'type': 'ineq',
                 'fun': lambda w: np.array([w[i] for i in range(len(w))])}]
        
        num_cols = input_np.shape[1]
        
        # 初始化权重也要是 float64
        init = np.array([1 / num_cols for i in range(num_cols)], dtype=np.float64)
        
        # 调用优化器
        result = opt.minimize(func, x0=init, constraints=cons, tol=1e-4)
        
        if not result.success:
            print("Optimization for BCRP not successful :")
            
        rel_rets = np.matmul(input_np, result.x)
        cum_rets = torch.cumprod(torch.Tensor(rel_rets), 0) # 结果转回 Tensor 以便后续兼容
        bcrp = torch.Tensor(result.x)
        bcrp_rets = cum_rets
        
        # 为了保持后续代码兼容，将 rel_rets 转回 Tensor 返回
        return torch.Tensor(rel_rets), bcrp_rets


def RiskMeasure(returns):
    # --- 关键修复开始 ---
    # 如果输入的 returns 张量在 GPU 上，必须先移动到 CPU 并切断梯度，
    # 否则无法转换为 numpy 数组供 pandas 和 empyrical 使用
    if returns.device.type != 'cpu':
        returns = returns.cpu().detach()
    # --- 关键修复结束 ---

    # 利用累积收益序列计算日收益率
    sr, cmr, sotr, mdd = torch.zeros(returns.size(1)), torch.zeros(returns.size(1)), torch.zeros(
        returns.size(1)), torch.zeros(returns.size(1))
        
    for s in range(returns.size(1)):
        # 此时 returns 已经在 CPU 上，可以安全转为 numpy
        daily_ret = torch.Tensor(np.array(pd.DataFrame(np.array(returns[:, s]))))-1
        daily_ret[0] = returns[0, s]-1
        
        # print(daily_ret)
        # 利用日收益率计算风险指标
        # 注意：empyrical 计算结果通常是 numpy float，转回 Tensor 以保持格式一致
        sr[s] = torch.tensor(empyrical.sharpe_ratio(daily_ret.numpy(), risk_free=0, period='daily'))
        cmr[s] = torch.tensor(empyrical.calmar_ratio(daily_ret.numpy(), period='daily'))
        sotr[s] = torch.tensor(empyrical.sortino_ratio(daily_ret.numpy(), required_return=0, period='daily',
                                                       _downside_risk=None))
        mdd[s] = torch.tensor(empyrical.max_drawdown(daily_ret.numpy()))
        
    return sr, cmr, sotr, mdd


def CyclePaint(df,lables):
      # print(lables)
      # 将日期转换为索引
      df.index = pd.to_datetime(df.index)

      # 平均序列的图
      plt.bar(lables.index, df, label=lables)
      plt.title("Original Data")
      plt.xlabel("Date")
      plt.ylabel("Value")
      plt.legend()
      plt.show()


      # 将序列转换成月度，并画出图
      df_monthly = df.groupby(df.index.month).mean()
      dates = [datetime.date(2023, month, 1) for month in range(1, 13)]
      plt.figure(figsize=(10, 5))
      plt.bar(dates, df_monthly, label=lables)
      plt.title("Monthly Data")
      plt.xlabel("Month")
      plt.ylabel("Value")
      plt.legend()
      plt.show()

      # 将序列转换成季度，并画出图
      df_quarterly = df.groupby(df.index.quarter).mean()
      dates = [datetime.date(2023, quarter, 1) for quarter in range(1, 5)]
      plt.figure(figsize=(10, 5))

      plt.bar(dates, df_quarterly, label=lables)
      plt.title("Quarterly Data")
      plt.xlabel("Quarter")
      plt.ylabel("Value")
      plt.legend()
      plt.show()


###分层函数定义###
class MIS(torch.nn.Module):  # asset i in window=L steps outputs
    def __init__(self, reg_size, ind_size, sto_size, hid, hid_sto, proj_size, attri):
        super(MIS, self).__init__()
        if proj_size == 0:
            out_size = hid
        else:
            out_size = proj_size
        self.hid = out_size
        self.lstmreg = torch.nn.LSTM(input_size=reg_size, hidden_size=int(hid/8), batch_first=True)
        self.lstmind = torch.nn.LSTM(input_size=ind_size+1, hidden_size=int(hid), batch_first=True) #+ 1
        self.lstmsto = torch.nn.LSTM(input_size=sto_size+n_i, hidden_size=hid_sto, batch_first=True)  # + n_i/+1
        self.lin_rnnm = torch.nn.Linear(int(hid/8), 1)
        self.lin_rnni = torch.nn.Linear(int(hid), 1)
        self.Att = torch.nn.MultiheadAttention(embed_dim=int(hid), num_heads=1, bias=True, kdim=hid, vdim=hid, batch_first=True)
        self.lin_rnns = torch.nn.Linear(hid, 1)
        self.sigmodm= torch.nn.Sigmoid()
        # self.sigmodi = torch.nn.Sigmoid()
        # self.softmax = torch.nn.Softmax(dim=0)
        self.ind_size = ind_size
        self.sto_size = sto_size
        self.hidi = int(hid)
        self.attri = attri

    def forward(self, reg_in, ind_in, sto_in):  # [N,L,Fr][N,L,I,Fi][N,L,S,Fs]
        # reg_in, ind_in, sto_in = data_input[0],data_input[1],data_input[2]
        ##**** Market layer ****##
        out_rm, _ = self.lstmreg(reg_in)
        out_r = self.sigmodm(self.lin_rnnm(out_rm))  # [N,L,1] 股价预测值

        # generate risk-free and risk allocation Normalize (N,L,2)
        # 将最后一个维度进行softmax归一化
        # f = torch.softmax(out_r, dim=-1)

        #**** Industry layer ****##
        ind_mar_in = torch.cat((ind_in[:, :, :, :], out_r.unsqueeze(-2).repeat(1, 1, ind_in.size(2), 1)),
                               dim=-1)  # [N,L,1,1]+[N,L,S,Fs]=[N,L,S,Fs+1]
        # ind_mar_in = ind_in[:, :, :, :]
        out_ii = torch.zeros(ind_in.size(0), ind_in.size(1), ind_in.size(2), self.hidi).to(device)
        att = torch.zeros(ind_in.size(0), ind_in.size(1), ind_in.size(2), self.hidi).to(device)
        for I in range(ind_mar_in.size(2)):
            out_ii[:, :, I, :], _ = self.lstmind(ind_mar_in[:, :, I, :])  # [N,L,1]---[N,L,I,1]
            att[:,:,I,:],attn_output_weights = self.Att(query=out_ii[:, :, I, :], key=out_ii[:, :, I, :], value=out_ii[:, :, I, :], need_weights=True, average_attn_weights=True)
        out_i = self.lin_rnni(att)  # [N,L,I,1]---[N,L,I,1]self.sigmodi()
        # # print(out_i.size())

        # ##**** Stock layer ****##
        # ##将每个stock的特征与score_i进行拼接
        sto_mar_in = torch.cat((sto_in, out_i.squeeze(3).unsqueeze(-2).repeat(1, 1, sto_in.size(2), 1)),
                               dim=-1)  # [N,L,S,1]+[N,L,S,Fs]=[N,L,S,Fs+1]
        # sto_mar_in = torch.cat((sto_in, out_r.unsqueeze(-2).repeat(1, 1, sto_in.size(2), 1)), dim=-1)#[N,L,S,1]+[N,L,S,Fs]=[N,L,S,Fs+1]
        # sto_mar_in = sto_in

        ##给出评分
        score_s = torch.zeros(sto_mar_in.size(0), sto_mar_in.size(2), sto_mar_in.size(1)).to(device)  # [N,A,L]
        # #Batch normalization layer
        # bns = torch.nn.BatchNorm2d(sto_in.size(0) * sto_in.size(1), sto_in.size(3)-1) # 改变输入张量的形状)
        for a in range(score_s.size(1)):
            out_sf, _ = self.lstmsto(sto_mar_in[:, :, a, :])
            # print(out_sf.size())
            out_s = self.lin_rnns(out_sf)  # [L,N,1]
            score_s[:, a, :] = out_s.squeeze(2)  # [N,A,L]

        if self.attri is None:
            return score_s  # [N,L],[N,A,L]
        else:
            print(score_s.size())
            return score_s.mean(0).mean(0)


#####数据载入处理SZ50######
####数据#####
# 设置随机种子
seed = 12
torch.manual_seed(seed)
# 使用一个固定的 meta-seed 保证这 30 个数每次运行是一样的，方便复现
# 如果你希望每次运行都完全不同，可以去掉 np.random.seed(42)
np.random.seed(seed) 
random_seeds = np.random.randint(1, 10000, size=30).tolist()
# 确保种子不重复 (虽大概率不会重复，但为了严谨)
random_seeds = list(set(random_seeds))
while len(random_seeds) < 30:
    new_seed = np.random.randint(1, 10000)
    if new_seed not in random_seeds:
        random_seeds.append(new_seed)
data_set = ['NIFTY50','CEMG']  # 'SSE50', 'SZSE100','NASDAQ','HS','NIFTY50','CEMG'
ablation = 3 #0:stock; 1:stock+industry; 2:stock+market; 3:stock+industry+market
mode = 4 # 0：训练；1：测试；2：可解释性分析；3：Optuna超参数优化；4：重复实验
lr, batch = 0.015, 64 #40 30, 0.02
early_stop_epochs = 5
tc = 0 #0, 0.0005, 0.001, 0.0015, 0.002
win, lambd, hidden = 20, 0.00001, 64 #0.00001
#10,20,40,80,160,320
#0, 0.000005, 0.00001, 0.000015, 0.00002, 0.000025
#16, 32, 64, 128, 256, 512

for data_name in data_set:
    print('Data:',data_name, )
    if data_name == 'SSE50':
        n_i, n_s, start = 27, 37,  1 #
    elif data_name == 'SZSE100':
        n_i, n_s, start = 27, 62, 950 #
    elif data_name == 'NASDAQ':
        n_i, n_s, start = 10, 73, 900 # 900
    elif data_name == 'HS':
        n_i, n_s, start = 12, 61, 1000 #
    elif data_name == 'CEMG':
        n_i, n_s, start = 6, 24, 40 #
    elif data_name == 'NIFTY50':
        n_i, n_s, start = 12, 47, 220 #220 epoch20
    
    
    if data_name in ['NIFTY50','CEMG']:
        f_m, f_i, f_s,  = 5, 4, 16  
        if data_name in ['NIFTY50']:
            data = torch.load(path + "/" + data_name + "_input.pt")[start:start+2000,:]
            rs = torch.load(path + "/" + data_name + "_return.pt")[1+start:1+start+2000, :] + 1
        else:
            data = torch.load(path + "/" + data_name + "_input.pt")[start:start+1000,:]
            rs = torch.load(path + "/" + data_name + "_return.pt")[1+start:1+start+1000, :] + 1
    else:
        f_m, f_i, f_s,  = 4, 4, 17 
        data = torch.from_numpy(pd.read_pickle(path + "/" + data_name + "_input_2012-01-01_2024-06-01.pkl").astype('float32').values[start:start+2000,:])
        rs = torch.tensor(pd.read_pickle(path + "/" + data_name + "_rss_2012-01-01_2024-06-01.pkl").values[1+start:1+start+2000, :].astype('float32')) + 1
    
    if data_name in ['CEMG']:
        train_len, test_len = 600, 400
    else:
        train_len, test_len = 1200, 800
    data_len = f_s * n_s 
    epoch = 30
    # data = torch.from_numpy(pd.read_pickle(path + "/" + data_name + "_input_2012-01-01_2024-06-01.pkl").astype('float32').values[start:start+2000,:])
    # rs = torch.tensor(pd.read_pickle(path + "/" + data_name + "_rss_2012-01-01_2024-06-01.pkl").values[1+start:1+start+2000, :].astype('float32')) + 1
    # print(data[0,-4:])
    #####数据划分######
    # 输入特征
    data_trn, data_ten = data[:train_len, ], data[train_len:train_len + test_len, ]
    # 输出对应的标签（相对价格）
    rs_tr, rs_te = rs[:train_len, ], rs[train_len:train_len + test_len, ]
    # 计算训练集的均值和标准差 
    data_train_mean = torch.mean(data_trn, dim=0)
    data_train_std = torch.std(data_trn, dim=0)

    # 归一化测试集
    data_tr = (data_trn - data_train_mean) / data_train_std
    data_te = (data_ten - data_train_mean) / data_train_std
    # print((data_train_std==0).sum().sum())

    # @title 模型参数初始化与数据载入
    nn0 = MIS(reg_size=f_m, ind_size=f_i, sto_size=f_s, hid=hidden, hid_sto=hidden, proj_size=1, attri=None).float().to(device)
    # 设置优化器
    adam = torch.optim.Adam(nn0.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=1e-03, amsgrad=False)
    # 数据样本生成
    data_all_w = Sampleset(data_tr, win)  # 拼接[L,f_m+f_i*I+f_s*S]
    datar_w = Sampleset(rs_tr, win)
    my_dataset = MydataM(data_all_w, datar_w)
    Obj = []
    Val_Obj = []
    time_start = time.time()
    if mode == 0:
        # 训练集训练
        best_obj = -float('inf')
        for e in range(epoch):
            nn0.train()
            # 使用DataLoader抽样
            train_loader = DataLoader(dataset=my_dataset, batch_size=batch, shuffle=True,
                                      drop_last=True)  # 是否要多线程，构成minibatch并行读取数据
            obj = torch.zeros(train_loader.__len__()).to(device)
            
            for it, (data_all, lables) in enumerate(train_loader):
                # 数据拆分
                data_m, data_i, data_s = data_all[:, :, -f_m:], \
                                         data_all[:, :, data_len:-f_m], \
                                         data_all[:, :, :data_len]  # [N,L,M*F][N,L,I*F][N,L,S*F]
                data_ms = data_m.reshape(data_m.size(0), win, f_m).to(device)
                # print(data_m.size(),data_i.size(),data_s.size(),data_all.size())
                data_is = data_i.reshape(data_i.size(0), win, int(data_i.size(2) / f_i), f_i).to(device)
                data_ss = data_s.reshape(data_s.size(0), win, int(data_s.size(2) / f_s), f_s).to(device)
                data_in = [data_ms, data_is, data_ss]
                # 市场层+行业层+资产层
                score = nn0.forward(reg_in=data_ms, ind_in=data_is, sto_in=data_ss)
                bc = PortfolioGe(score)
                # 结合风险资产与无风险资产的收益，构建损失值，反向传播更新网络
                # sharpe = Object(bc, (lables.to(device)).transpose(1, 2)) - lambd * torch.norm(bc, p=2, dim=1,
                #                                                                                 keepdim=False, out=None,
                #                                                                                 dtype=None).mean()
                # 确保在你的参数设置区域设置了一个合理的 tc，例如 tc = 0.0015
                # 调用 Object 时传入 tc
                sharpe = Object(bc, (lables.to(device)).transpose(1, 2), tc) - lambd * torch.norm(bc, p=2, dim=1, keepdim=False).mean()
                ob = sharpe  # (torch.sqrt(lossr*lossi)/N,L/N,L/N,L---N,L----
                adam.zero_grad()
                (-ob).backward()
                adam.step()
                obj[it] = float(ob)
            Obj.append(float(obj.mean()))
            torch.save(nn0.state_dict(), path1 + '/mis' + '_' + data_name + '_' + 'win' +
                       str(win) + 'lr' + str(lr) + 'ep' +
                       str(epoch) +'lam'+str(lambd)+ 'ab'+str(ablation)+'hid'+str(hidden)+'tc'+str(tc)+'fd.pth')  # 保存模型参数
            time_2 = time.time()
            # 记录当前 Epoch 的平均 Loss (这里其实是 Sharpe Ratio 的相关值)
            current_epoch_metric = float(obj.mean())
            Obj.append(current_epoch_metric)
            print("epoch:", e, "finished.", '\n', "Loss:", Obj[e], "time:", time_2 - time_start)
            # scheduler.step(current_epoch_metric)



    elif mode == 1:
        # 测试集
        device = 'cpu' # 建议在CPU上进行评估
        # 加载模型 (文件名使用训练时设定的全局 tc)
        nn1 = MIS(reg_size=f_m, ind_size=f_i, sto_size=f_s, hid=hidden, hid_sto=hidden, proj_size=1, attri=None).float().to(device)
        nn1.load_state_dict(torch.load(path1 + '/mis' + '_' + data_name + '_' + 'win' +
                                       str(win) + 'lr' + str(lr) + 'ep' +
                                       str(epoch) + 'lam'+str(lambd)+'ab'+str(ablation)+'hid'+str(hidden)+'tc'+str(tc)+'fd.pth', map_location=device))
        
        with torch.no_grad():
            nn1.eval()
            # 计算累积收益 (生成权重 BC_test)
            BC_test = torch.zeros(int(test_len / win) * win, n_s).to(device)
            for s in range(int(test_len / win)):
                test_t = data_te[s * win:(s + 1) * win, :].unsqueeze(0)
                data_mt, data_it, data_st = test_t[:, :, -f_m:], \
                                            test_t[:, :, data_len:-f_m].reshape(1, win, n_i, f_i), \
                                            test_t[:, :, :data_len].reshape(1, win, n_s, f_s) 
                data_inte = [data_mt.to(device), data_it.to(device), data_st.to(device)]
                # 市场层+行业层+资产层
                scoret = nn1.forward(reg_in=data_mt, ind_in=data_it, sto_in=data_st)
                bct = PortfolioGe(scoret)
                BC_test[s * win:(s + 1) * win, ] = bct.squeeze(0).transpose(0, 1)

            ##计算基准策略##
            rp_batchm = rs_te[: int(test_len / win) * win, :]
            CRim, CRm = Market(rp_batchm)
            CRibd, CRib = Best(rp_batchm)
            CRibcd, CRibc = BCRP.run(rp_batchm)
            
            # ================= 敏感性分析开始 =================
            tc_sensitivity_list = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]
            
            print("\n" + "="*80)
            print(f"Transaction Cost Sensitivity Analysis (Model trained with tc={tc})")
            print("-" * 80)
            print(f"{'Test TC':<12} | {'Final CW':<12} | {'Sharpe':<12} | {'Calmar':<12} | {'MDD':<12}")
            print("-" * 80)
            
            for test_tc in tc_sensitivity_list:
                # 使用当前的 test_tc 计算净值曲线
                # 注意：BC_test (权重) 不变，只改变扣费比例
                CR_test_sens = Return_test(BC_test, rp_batchm, test_tc)
                CRF_sens = torch.cumprod(CR_test_sens, dim=1).squeeze(0)
                
                # 计算风险指标
                sr_s, cmr_s, sotr_s, mdd_s = RiskMeasure(CR_test_sens.reshape(int(test_len / win) * win, 1))
                
                # 提取标量值以便打印
                final_cw = CRF_sens[-1].item() if isinstance(CRF_sens[-1], torch.Tensor) else CRF_sens[-1]
                v_sr = sr_s.item() if isinstance(sr_s, torch.Tensor) else sr_s
                v_cmr = cmr_s.item() if isinstance(cmr_s, torch.Tensor) else cmr_s
                v_mdd = mdd_s.item() if isinstance(mdd_s, torch.Tensor) else mdd_s

                print(f"{test_tc:<12} | {final_cw:<12.4f} | {v_sr:<12.4f} | {v_cmr:<12.4f} | {v_mdd:<12.4f}")
            print("="*80 + "\n")
            # ================= 敏感性分析结束 =================

            # 以下为原有的绘图逻辑（使用全局 tc 变量，作为主结果展示）
            CR_testc = Return_test(BC_test, rp_batchm, tc)
            CRF = torch.cumprod(CR_testc, dim=1).squeeze(0)

            # 画图查看样本外表现
            if isinstance(CRF, np.ndarray) and CRF.dtype == np.float32 and CRF.flags['C_CONTIGUOUS']:
                # 数据在GPU上
                CRF = CRF.get()  # 将数据从GPU转移到CPU
            plt.figure()
            plt.subplot(111)
            plt.plot(np.arange(0, int(test_len / win) * win), CRF, 'b-', label=f'LSTM (tc={tc})') 
            plt.plot(np.arange(0, int(test_len / win) * win), CRm.numpy(), 'r-', label='Maket')
            plt.plot(np.arange(0, int(test_len / win) * win), CRib.numpy(), 'g-', label='Best')
            plt.plot(np.arange(0, int(test_len / win) * win), CRibc.numpy(), 'y-', label='BCRP')
            plt.xlabel('day')
            plt.ylabel('CR')
            plt.legend()
            plt.title(f'Test Set Performance (Base tc={tc})')
            plt.show()
            
            sr1, cmr1, sotr1, mdd1 = RiskMeasure(CRim.reshape(int(test_len / win) * win, 1) )
            sr2, cmr2, sotr2, mdd2 = RiskMeasure(CRibd.reshape(int(test_len / win) * win, 1) )
            sr3, cmr3, sotr3, mdd3 = RiskMeasure(CRibcd.reshape(int(test_len / win) * win, 1) )
            sr4, cmr4, sotr4, mdd4 = RiskMeasure(CR_testc.reshape(int(test_len / win) * win, 1) )
            
            print('BCRP:', CRibc[-1], 'BEST:', CRib[-1], 'MARKET:', CRm[-1], 'LSTM:', CRF[-1])
            print('M:Sharpe,CMR,SOTR,MDD:', sr1, cmr1, sotr1, mdd1)
            print('Best:Sharpe,CMR,SOTR,MDD:', sr2, cmr2, sotr2, mdd2)
            print('BCRP:Sharpe,CMR,SOTR,MDD:', sr3, cmr3, sotr3, mdd3)
            print('LSTM:Sharpe,CMR,SOTR,MDD:', sr4, cmr4, sotr4, mdd4)
            
            # 计算训练集累积收益 (此处逻辑保持不变)
            # ... (后续训练集收益计算代码略，保持原样即可)
            BC_train = torch.zeros(int(train_len / win) * win, n_s).to(device)
            for s in range(int(train_len / win)):
                train_t = data_tr[s * win:(s + 1) * win, :].unsqueeze(0)
                rs_trr = rs_tr[s * win:(s + 1) * win, :].unsqueeze(0) 
                data_mtr, data_itr, data_str = train_t[:, :, -f_m:], \
                                               train_t[:, :, data_len:-f_m].reshape(1, win, n_i, f_i), \
                                               train_t[:, :, :data_len].reshape(1, win, n_s, f_s) 
                
                scoretr = nn1.forward(reg_in=data_mtr.to(device), ind_in=data_itr.to(device), sto_in=data_str.to(device))
                bctr = PortfolioGe(scoretr)
                BC_train[s * win:(s + 1) * win, ] = bctr.squeeze(0).transpose(0, 1)

            rp_batchmtr = rs_tr[: int(train_len / win) * win, :]
            CR_train = Return_test(BC_train, rp_batchmtr, tc)
            CRFt = torch.cumprod(CR_train, dim=1).squeeze(0)



    elif mode == 2:
        # 创建模型实例和积分梯度对象
        model = MIS(reg_size=f_m, ind_size=f_i, sto_size=f_s, hid=hidden,
                    hid_sto=hidden, proj_size=1, attri=1).float().to(device)
        model.load_state_dict(torch.load(path1 + '/mis' + '_' + data_name + '_' + 'win' +
                                         str(win) + 'lr' + str(lr) + 'ep' +
                                         str(epoch) + 'lam'+str(lambd)+'ab'+str(ablation)+'hid'+str(hidden)+'tc'+str(tc)+'fd.pth', map_location=device))

        ig = IntegratedGradients(model)
        # 构建目标数据
        target_data = (data_te[:, -f_m:].unsqueeze(0).to(device),
                       data_te[:, data_len:-f_m].unsqueeze(0).reshape(1, win * int(test_len / win), n_i, f_i).to(device),
                       data_te[:, :data_len].unsqueeze(0).reshape(1, win * int(test_len / win), n_s, f_s).to(device))
        # print(model(target_data))
        print('target1', target_data[0].size())
        print('target2', target_data[1].size())
        print('target3', target_data[2].size())

        # # 计算特征重要性
        attributions = ig.attribute(target_data, n_steps=20)  # 不需要target，因为only returns a scalar value per example

        # # 打印特征重要性
        print('Market:', attributions[0].size())
        print('Sector:', attributions[1].size())
        print('Stock:', attributions[2].size())
        Marketf, Markett = attributions[0].squeeze(), attributions[0].squeeze().sum(-1).unsqueeze(-1)
        Sectori = attributions[1].squeeze().mean(-1)
        Stockf = attributions[2].squeeze().mean(-2)
        # print('Marketf:',  Marketf.size())#[800,4]
        # print('Markett:',  Markett.size())#[800,1]
        # print('Sectori:',  Sectori.size())#[800,27]
        # print('Stockf:',  Stockf.size())#[800,37]
        # print(Markett)

        # 获取列名
        lable_i_SSE50 = ['agriculture','basic chemical','ferrous metal','nonferrous metal','electronics','household appliance','food & beverage','textile & apparel','light-industry manufacturing','medical biology',
                'utilities','transportation','real estate','commerce','leisure service','conglomerate','building material','architectural ornament','electrical equipment','defence & military','computer','media',
                'telecommunication','bank','non-bank finance','automobile','equipment']
        lable_i_SZSE100 = lable_i_SSE50
        lable_i_NASDAQ = ['bank','biotechnology','computer','healthcare','industrial','insurance','internet','other finance','transportation','telecommunication']
        lable_i_HS = ['energy','info-tech','conglomerates','materials','industrials','consumer discretionary','consumer staples','healthcare','telecommunications','utilities','financials','properties & construction']
        lable_i = { 'lable_i_SSE50': lable_i_SSE50, 'lable_i_SZSE100': lable_i_SZSE100, 'lable_i_NASDAQ': lable_i_NASDAQ, 'lable_i_HS': lable_i_HS}
        lable_s = ['op', 'cl', 'hp', 'lp', 'vl', 'to','cpt', 'ma5', 'ma10',
                   'vema5', 'vema10', 'rsi', 'rstr12', 'rstr24', 'revs5',
                   'revs10', 'revs20']

        mark = ['.','v','s','*','+','|']
        lable_m = ['$op_m$','$cp_m$','$hp_m$','$lp_m$']#['op', 'cl', 'hp', 'lp']
        colors = ['silver','r','g','y','k','c']

        # # 特征
        # Step 2: Split the tensor into 4 groups of 20 elements each
        marf = torch.stack([Marketf.cpu()[i:i+win, :] for i in range(0, 800, win)], dim=1).mean(1).squeeze(1)
        mart = torch.stack([Markett.cpu()[i:i+win, :] for i in range(0, 800, win)], dim=1).mean(1).squeeze(1)
        print(marf.size(),mart.size())
        # marf = Marketf.cpu().reshape(20,-1,4).mean(1).squeeze()#[20,4] 
        # mart = Markett.cpu().reshape(20,-1,1).mean(1).squeeze()#[20]
        ind = Sectori.cpu().mean(0).unsqueeze(0)#[1,n_i]
        sto = Stockf.cpu().mean(0).unsqueeze(0)#[1,n_s] 
        # print('market',marf,mart,'ind',ind,'sto',sto)
        print('market',marf.size(),mart.size(),'ind',ind.size(),'sto',sto.size())


        Marf = pd.DataFrame(marf, columns=lable_m)#, index=dates
        Mart = pd.DataFrame(mart, columns=['composite attribution'])#, index=dates
        Ind = pd.DataFrame(ind, columns=lable_i[f'lable_i_{data_name}'])#, index=dates
        Sto = pd.DataFrame(sto, columns=lable_s)#, index=dates
        # print('market',Marf,Mart,'ind',Ind,'sto',Sto)
        
        
        ## 根据尺度画图
        #市场win=20期f = 4+1
        # 绘制折线图
        plt.figure(figsize=(8, 2))
        for i in range(4):
            plt.plot(range(1,win+1), marf[:,i], marker=mark[i], color=colors[i], label=lable_m[i])
        # 绘制条形图
        plt.bar(range(1,21), mart, label='composite \nattribution', alpha=0.5)
        # 添加标题和坐标轴标
        plt.xlabel("Trading periods")#Month,Industry
        plt.ylabel(r'$\bar{\delta}_{i^{\prime}}$')
        # plt.ylabel(r'$\mathrm{bar{\delta}_{M,i}}$')
        plt.legend(ncol=3,prop={'size': 8})#prop={'size':9},
        # 自动调整横坐标标签的显示
        plt.xlim(0.5,20.5)
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))
        # 显示图形
        plt.grid(True,linestyle="--",color="gray",linewidth="0.2",axis="both")
        # 添加y=0的线
        plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
        plt.savefig('XAI_m'+str(data_name)+'.eps',format='eps', dpi=1000, bbox_inches = 'tight')
        plt.show()

    elif mode == 3:
        # ================== 新增：Optuna 超参数优化 ==================
        print(f"Starting Optuna optimization for {data_name}...")
        
        def objective(trial):
            # 1. 建议超参数
            trial_lr = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
            trial_win = trial.suggest_categorical('win', [10, 20, 40, 80, 160, 320])
            trial_hidden = trial.suggest_categorical('hidden', [16, 32, 64, 128, 256, 512])
            trial_lambd = trial.suggest_categorical('lambd', [0, 5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5])
            
            # 2. 重新准备数据 (因为 win 变了，所以 SampleSet 必须重新运行)
            # 注意：此处不修改全局变量，而是创建局部变量
            data_all_w_trial = Sampleset(data_tr, trial_win)
            datar_w_trial = Sampleset(rs_tr, trial_win)
            my_dataset_trial = MydataM(data_all_w_trial, datar_w_trial)
            
            train_loader_trial = DataLoader(dataset=my_dataset_trial, batch_size=batch, shuffle=True, drop_last=True)
            
            # 3. 初始化模型
            model_trial = MIS(reg_size=f_m, ind_size=f_i, sto_size=f_s, hid=trial_hidden, hid_sto=trial_hidden, proj_size=1, attri=None).float().to(device)
            optimizer = torch.optim.Adam(model_trial.parameters(), lr=trial_lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=1e-03, amsgrad=False)
            
            # 4. 训练循环 (简化版，无打印，无保存)
            for e in range(epoch):
                model_trial.train()
                for data_all, lables in train_loader_trial:
                    data_m, data_i, data_s = data_all[:, :, -f_m:], \
                                             data_all[:, :, data_len:-f_m], \
                                             data_all[:, :, :data_len]
                    
                    data_ms = data_m.reshape(data_m.size(0), trial_win, f_m).to(device)
                    data_is = data_i.reshape(data_i.size(0), trial_win, int(data_i.size(2) / f_i), f_i).to(device)
                    data_ss = data_s.reshape(data_s.size(0), trial_win, int(data_s.size(2) / f_s), f_s).to(device)
                    
                    score = model_trial(reg_in=data_ms, ind_in=data_is, sto_in=data_ss)
                    bc = PortfolioGe(score)
                    
                    sharpe = Object(bc, (lables.to(device)).transpose(1, 2), tc) - trial_lambd * torch.norm(bc, p=2, dim=1, keepdim=False).mean()
                    ob = sharpe
                    
                    optimizer.zero_grad()
                    (-ob).backward()
                    optimizer.step()
            
            # 5. 验证/测试评估 (使用 Test Set 计算 Sharpe Ratio 作为优化目标)
            # 注意：为了寻找泛化性能最好的参数，这里暂时使用测试集表现作为反馈信号
            with torch.no_grad():
                model_trial.eval()
                # 重新计算测试集需要的步数
                steps = int(test_len / trial_win)
                BC_test_trial = torch.zeros(steps * trial_win, n_s).to(device)
                
                for s in range(steps):
                    test_t = data_te[s * trial_win:(s + 1) * trial_win, :].unsqueeze(0)
                    data_mt, data_it, data_st = test_t[:, :, -f_m:], \
                                                test_t[:, :, data_len:-f_m].reshape(1, trial_win, n_i, f_i), \
                                                test_t[:, :, :data_len].reshape(1, trial_win, n_s, f_s)
                    
                    scoret = model_trial(reg_in=data_mt.to(device), ind_in=data_it.to(device), sto_in=data_st.to(device))
                    bct = PortfolioGe(scoret)
                    BC_test_trial[s * trial_win:(s + 1) * trial_win, ] = bct.squeeze(0).transpose(0, 1)
                
                # --- 修改：明确将验证集收益数据放入 device ---
                rp_batchm = rs_te[: steps * trial_win, :].to(device) 
                
                # 调用 Return_test (此时输入都在 GPU 上了)
                CR_testc = Return_test(BC_test_trial, rp_batchm, tc)
                
                # 计算 Sharpe Ratio
                sr_trial, _, _, _ = RiskMeasure(CR_testc.reshape(steps * trial_win, 1))
                

                # === 清理显存 ===
                del model_trial
                del optimizer
                del bc, score
                del data_ms, data_is, data_ss
                # 强制执行垃圾回收
                import gc
                gc.collect()
                # 清空 PyTorch 的 CUDA 缓存
                torch.cuda.empty_cache()


                return sr_trial.mean().item()



        # 创建 Study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20) # 试验次数可根据需要调整
        
        print(f"Best params for {data_name}:")
        print(study.best_params)
        print("Best Sharpe:", study.best_value)
        print("-" * 30)



    elif mode == 4:
        # ================== 修正版：多次实验统计显著性测试 ==================
        print(f"\n{'='*20} Running Statistical Significance Test (30 Random Seeds) for {data_name} {'='*20}")
        
        n_runs = len(random_seeds)
        print(f"Generated Seeds: {random_seeds}")
        
        metrics_log = {'Final_CW': [], 'Sharpe': [], 'Calmar': [], 'MDD': []}
        excess_returns_pool = [] 
        market_metrics = {}
        
        # 初始化字典，用于存储当前数据集下所有 Seed 的日收益率序列
        seed_returns_dict = {} 
        
        for run_idx, seed_i in enumerate(random_seeds):
            print(f"\n>>> Run {run_idx + 1}/{n_runs} | Seed: {seed_i}")
            
            # 设置当前随机种子
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_i)
            
            # 路径
            seed_model_name = f'mis_{data_name}_win{win}lr{lr}ep{epoch}lam{lambd}ab{ablation}hid{hidden}tc{tc}fd_seed{seed_i}.pth'
            seed_model_path = os.path.join(path1, seed_model_name)
            
            # 初始化模型
            model = MIS(reg_size=f_m, ind_size=f_i, sto_size=f_s, hid=hidden, 
                        hid_sto=hidden, proj_size=1, attri=None).float().to(device)
            
            # --- 检查或训练 ---
            if os.path.exists(seed_model_path):
                print(f"[*] Found checkpoint: {seed_model_name}")
                # 增加 weights_only=False 以防部分旧版 pytorch 警告
                model.load_state_dict(torch.load(seed_model_path, map_location=device))
            else:
                print(f"[!] Training from scratch...")
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), 
                                             eps=1e-06, weight_decay=1e-03, amsgrad=False)
                
                data_all_w_run = Sampleset(data_tr, win)
                datar_w_run = Sampleset(rs_tr, win)
                dataset_run = MydataM(data_all_w_run, datar_w_run)
                train_loader = DataLoader(dataset=dataset_run, batch_size=batch, shuffle=True, drop_last=True)
                
                for e in range(epoch):
                    model.train()
                    for data_all, lables in train_loader:
                        data_m, data_i, data_s = data_all[:, :, -f_m:], \
                                                 data_all[:, :, data_len:-f_m], \
                                                 data_all[:, :, :data_len]
                        
                        data_ms = data_m.reshape(data_m.size(0), win, f_m).to(device)
                        data_is = data_i.reshape(data_i.size(0), win, int(data_i.size(2) / f_i), f_i).to(device)
                        data_ss = data_s.reshape(data_s.size(0), win, int(data_s.size(2) / f_s), f_s).to(device)
                        
                        score = model(reg_in=data_ms, ind_in=data_is, sto_in=data_ss)
                        bc = PortfolioGe(score)
                        
                        # 使用修正后的 Object 函数 (含 clamp)
                        sharpe = Object(bc, (lables.to(device)).transpose(1, 2), tc) - lambd * torch.norm(bc, p=2, dim=1).mean()
                        ob = sharpe
                        
                        optimizer.zero_grad()
                        (-ob).backward()
                        optimizer.step()
                
                torch.save(model.state_dict(), seed_model_path)
            
            # --- 测试阶段 ---
            model.eval()
            with torch.no_grad():
                steps = int(test_len / win)
                BC_test = torch.zeros(steps * win, n_s).to(device)
                
                for s in range(steps):
                    test_t = data_te[s * win:(s + 1) * win, :].unsqueeze(0)
                    
                    data_mt, data_it, data_st = test_t[:, :, -f_m:], \
                                                test_t[:, :, data_len:-f_m].reshape(1, win, n_i, f_i), \
                                                test_t[:, :, :data_len].reshape(1, win, n_s, f_s)
                    
                    scoret = model(reg_in=data_mt.to(device), ind_in=data_it.to(device), sto_in=data_st.to(device))
                    bct = PortfolioGe(scoret)
                    BC_test[s * win:(s + 1) * win, ] = bct.squeeze(0).transpose(0, 1)
                
                rp_batchm = rs_te[: steps * win, :].to(device)
                
                # 1. 计算策略日收益率 (R_strategy, 这里是 1+r)
                daily_ret_strategy_gross = Return_test(BC_test, rp_batchm, tc).squeeze(0) # [T]
                
                # 保存日收益 (转为 numpy CPU)
                seed_returns_dict[seed_i] = daily_ret_strategy_gross.cpu().detach().numpy()

                # 2. 计算基准日收益率 (R_market)
                if run_idx == 0:
                    daily_ret_market_gross, _ = Market(rp_batchm.cpu())
                    daily_ret_market_gross = daily_ret_market_gross.to(device) # [T]
                    
                    # 注意：RiskMeasure 接受 1+r
                    msr, mcmr, _, mmdd = RiskMeasure(daily_ret_market_gross.unsqueeze(1))
                    market_metrics = {
                        'Final_CW': torch.cumprod(daily_ret_market_gross, dim=0)[-1].item(),
                        'Sharpe': msr.item(), 'MDD': mmdd.item()
                    }
                    seed_returns_dict['Market_Benchmark'] = daily_ret_market_gross.cpu().detach().numpy()
                
                # --- 记录指标 ---
                # 计算最终财富 CW
                CRF = torch.cumprod(daily_ret_strategy_gross, dim=0)
                final_cw = CRF[-1].item()
                
                # [关键修正] 传递日收益(1+r)给 RiskMeasure，而不是 CRF
                sr, cmr, _, mdd = RiskMeasure(daily_ret_strategy_gross.unsqueeze(1))
                
                metrics_log['Final_CW'].append(final_cw)
                metrics_log['Sharpe'].append(sr.item())
                metrics_log['Calmar'].append(cmr.item())
                metrics_log['MDD'].append(mdd.item())
                
                # 记录超额收益 (Simple Returns 差值)
                # 策略(1+r) - 市场(1+r) = 策略r - 市场r
                excess_ret = daily_ret_strategy_gross - daily_ret_market_gross
                excess_returns_pool.append(excess_ret.cpu().numpy())
                
                print(f"  Result: CW={final_cw:.4f}, SR={sr.item():.4f}")

        # --- 4. 统计报告 ---
        print(f"\n{'='*20} Statistical Report ({data_name} - 30 Runs) {'='*20}")
        for metric, values in metrics_log.items():
            print(f"{metric:<10}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
        wins = sum([cw > market_metrics['Final_CW'] for cw in metrics_log['Final_CW']])
        print(f"Win Rate (vs Market): {wins/n_runs*100:.1f}% ({wins}/{n_runs})")
        
        # t-test
        avg_excess_returns = [np.mean(er) for er in excess_returns_pool]
        t_stat, p_val = scipy.stats.ttest_1samp(avg_excess_returns, 0, alternative='greater')
        print(f"T-Test (Excess > 0): t={t_stat:.4f}, p={p_val:.4e}")
        if p_val < 0.05:
            print("  => Result is Statistically Significant (p < 0.05)")
        else:
            print("  => Result is NOT Statistically Significant")
            
        # 保存所有 Seed 的日收益率
        df_seed_returns = pd.DataFrame(seed_returns_dict)
        save_file_name = f'{data_name}_daily_returns_30seeds.pkl'
        save_full_path = os.path.join(path1, save_file_name)
        
        df_seed_returns.to_pickle(save_full_path)
        print(f"\n[Saved] All seeds daily returns saved to: {save_full_path}")
        print("-" * 60)