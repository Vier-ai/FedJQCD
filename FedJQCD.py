import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import argparse
import swanlab
import json
from collections import Counter
import time


# ======================
# Config (defaults, will be overridden by argparse)
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 10
ROUNDS = 100
LOCAL_EPOCHS = 15
BATCH_SIZE = 64

ETA_LOCAL = 0.02
ETA_GLOBAL = 1.0

DATASET = "MNIST"
NONIID = False
NONIID_ALPHA = 0.5
C = 1

SAVE_DIR = "figures"
FULL_PREC_BITS = 32

# ======================
# Model
# ======================
class CNN(nn.Module):
    def __init__(self, dataset="MNIST"):
        super().__init__()
        if dataset == "MNIST":
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 28, 28)
                dummy_out = self.features(dummy)
                fc_in = dummy_out.shape[1]
            self.classifier = nn.Linear(fc_in, 10)
        elif dataset == "CIFAR10":
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten()
            )
            self.classifier = nn.Linear(128, 10)
        self.net = nn.Sequential(self.features, self.classifier)

    def forward(self, x):
        return self.net(x)

# ======================
# Utils
# ======================
def get_flat(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat(model, flat):
    idx = 0
    for p in model.parameters():
        num = p.numel()
        p.data = flat[idx:idx + num].view_as(p).clone()
        idx += num

# ======================
# Quantization
# ======================
def quantize(x, bits=8, group_size=-1, stochastic=True):
    """
    非对称量化 + 可选随机舍入，支持分组。
    x: 任意形状张量
    bits: 量化位数（不包括符号位，实际表示范围为 [0, 2^bits - 1]）
    group_size: 分组大小，-1 表示全局一个 scale/zero
    stochastic: 是否使用随机舍入（无偏）
    返回:
        q_int: 量化后的整数张量，dtype=torch.int8 或 torch.int16
        scale: 缩放因子（张量或标量）
        zero: 零点（张量或标量）
    """
    if group_size > 0:
        original_shape = x.shape
        x_flat = x.reshape(-1)
        total_len = x_flat.shape[0]
        num_groups = (total_len + group_size - 1) // group_size
        q_levels = (1 << bits) - 1
        
        q_int_list = []
        scale_list = []
        zero_list = []
        
        for i in range(num_groups):
            start = i * group_size
            end = min(start + group_size, total_len)
            group = x_flat[start:end]
            min_val = group.min().item()
            max_val = group.max().item()
            
            if min_val == max_val:
                scale = 1.0
                zero = 0.0
                q_group = torch.zeros_like(group, dtype=torch.int8 if bits <= 8 else torch.int16)
            else:
                scale = (max_val - min_val) / q_levels
                zero = -min_val / scale
                float_q = group / scale + zero
                if stochastic:
                    floor = torch.floor(float_q)
                    prob = float_q - floor
                    mask = torch.rand_like(prob) < prob
                    q_group = (floor + mask).clamp(0, q_levels).to(torch.int8 if bits <= 8 else torch.int16)
                else:
                    q_group = torch.round(float_q).clamp(0, q_levels).to(torch.int8 if bits <= 8 else torch.int16)
            
            q_int_list.append(q_group)
            scale_list.append(scale)
            zero_list.append(zero)
        
        q_int_flat = torch.cat(q_int_list, dim=0)
        q_int = q_int_flat.reshape(original_shape)
        scale = torch.tensor(scale_list, device=x.device, dtype=torch.float32)
        zero = torch.tensor(zero_list, device=x.device, dtype=torch.float32)
        return q_int, scale, zero
    else:
        min_val = x.min().item()
        max_val = x.max().item()
        q_levels = (1 << bits) - 1
        
        if min_val == max_val:
            scale = torch.tensor(1.0, device=x.device)
            zero = torch.tensor(0.0, device=x.device)
            q_int = torch.zeros_like(x, dtype=torch.int8 if bits <= 8 else torch.int16)
        else:
            scale = (max_val - min_val) / q_levels
            zero = -min_val / scale
            float_q = x / scale + zero
            if stochastic:
                floor = torch.floor(float_q)
                prob = float_q - floor
                mask = torch.rand_like(prob) < prob
                q_int = (floor + mask).clamp(0, q_levels).to(torch.int8 if bits <= 8 else torch.int16)
            else:
                q_int = torch.round(float_q).clamp(0, q_levels).to(torch.int8 if bits <= 8 else torch.int16)
        
        return q_int, scale, zero

def dequantize(q_int, scale, zero):
    return (q_int.float() - zero) * scale

# ======================
# Data Split
# ======================
def split_dataset(dataset, num_clients, noniid=False, alpha=0.5):
    if not noniid:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        data_per_client = len(dataset) // num_clients
        client_data = []
        for i in range(num_clients):
            idx = indices[i * data_per_client:(i + 1) * data_per_client]
            client_data.append(torch.utils.data.Subset(dataset, idx))
        return client_data
    else:
        num_classes = 10
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        class_indices = [np.where(labels == k)[0] for k in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            prop = np.random.dirichlet(np.repeat(alpha, num_clients))
            num_samples_per_client = (prop * len(class_indices[c])).astype(int)
            num_samples_per_client[-1] = len(class_indices[c]) - num_samples_per_client[:-1].sum()
            np.random.shuffle(class_indices[c])
            start = 0
            for k in range(num_clients):
                end = start + num_samples_per_client[k]
                client_indices[k].extend(class_indices[c][start:end])
                start = end
        client_data = [torch.utils.data.Subset(dataset, idx) for idx in client_indices]
        return client_data

# ======================
# Client
# ======================
class Client:
    def __init__(self, model, dataset,
                 use_cv=True, use_quant=True, use_ef=True,
                 ef_variant="residual",
                 ef_beta=0.9,
                 ef_clip=1.0,
                 quant_bits=8,
                 ef_decay_factor=1.0, class_weights=None,
                 quant_type="uniform", group_size=-1,
                 quant_control=False, ef_control=False, quant_control_bits=8, quant_control_group_size=-1):
        self.model = copy.deepcopy(model).to(DEVICE)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.use_cv = use_cv
        self.use_quant = use_quant
        self.use_ef = use_ef
        self.quant_bits = quant_bits  
        self.class_weights = class_weights
        self.quant_type = quant_type
        self.ef_variant = ef_variant
        self.ef_beta = ef_beta
        self.ef_clip = ef_clip
        self.ef_decay_factor = ef_decay_factor
        self.group_size = group_size   

        # 新增：控制变量量化相关
        self.quant_control = quant_control
        self.ef_control = ef_control
        self.quant_control_bits = quant_control_bits
        self.quant_control_group_size = quant_control_group_size
        
        self.c_i = torch.zeros_like(get_flat(self.model)).to(DEVICE) if use_cv else None
        self.e_i = torch.zeros_like(get_flat(self.model)).to(DEVICE) if use_ef else None
        # 控制变量的误差反馈项
        self.ef_c = torch.zeros_like(get_flat(self.model)).to(DEVICE) if (use_cv and quant_control and ef_control) else None

    def _quantize_tensor(self, x, bits=None, group_size=None):
        """对张量进行量化并反量化，返回有损版本"""
        if bits is None:
            bits = self.quant_bits
        if group_size is None:
            group_size = self.group_size
        q_int, scale, zero = quantize(x, bits=bits, group_size=group_size, stochastic=True)
        return dequantize(q_int, scale, zero)
    
    def _quantize_and_feedback(self, delta, ef_state, bits, group_size, use_ef):
        """通用的量化+误差反馈处理，返回量化后的delta和新的ef_state"""
        if not use_ef:
            # 直接量化
            delta_hat = self._quantize_tensor(delta, bits=bits, group_size=group_size)
            residual = delta - delta_hat
            return delta_hat, residual, residual  # 残差用于更新ef
        else:
            # 先加误差缓存
            u = delta + ef_state
            u_hat = self._quantize_tensor(u, bits=bits, group_size=group_size)
            residual = u - u_hat  # 当前量化残差
            # 更新ef（根据变体）
            if self.ef_variant == "residual":
                new_ef = residual
            elif self.ef_variant == "momentum":
                new_ef = self.ef_beta * ef_state + residual
            elif self.ef_variant == "normalized":
                norm_val = torch.norm(u) + 1e-8
                new_ef = residual / norm_val
            elif self.ef_variant == "clipped":
                new_ef = torch.clamp(residual, -self.ef_clip, self.ef_clip)
            else:
                new_ef = residual
            new_ef = self.ef_decay_factor * new_ef
            return u_hat, new_ef, residual

    def train(self, w_global, c_global=None):
        set_flat(self.model, w_global)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(DEVICE) if self.class_weights is not None else None)
        
        total_loss = 0.0
        count = 0
        actual_steps = 0

        # 模拟服务器下发 c_global 的量化（如果启用控制变量量化）
        if self.use_cv and c_global is not None and self.quant_control:
            # 量化再反量化，模拟有损传输
            c_global = self._quantize_tensor(c_global, bits=self.quant_control_bits, group_size=self.quant_control_group_size)
        
        # correction
        if self.use_cv and c_global is not None:
            correction_flat = c_global - self.c_i
        else:
            correction_flat = torch.zeros_like(w_global).to(DEVICE)

        for _ in range(LOCAL_EPOCHS):
            for x, y in self.loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = self.model(x)
                if out.dim() == 3:
                    B, T, V = out.shape
                    loss = loss_fn(out.reshape(B*T, V), y.reshape(B*T))
                elif out.dim() == 2:
                    loss = loss_fn(out, y)
                else:
                    raise ValueError(f"Unexpected output dimension: {out.dim()}")
                loss.backward()
                with torch.no_grad():
                    idx = 0
                    for p in self.model.parameters():
                        num = p.numel()
                        corr = correction_flat[idx:idx+num].view_as(p)
                        p.data -= ETA_LOCAL * (p.grad + corr)
                        idx += num
                self.model.zero_grad()
                actual_steps += 1
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count
        w_local = get_flat(self.model)
        delta = w_local - w_global

        # ======================
        # 对模型更新 delta 进行量化 + EF（原有逻辑）
        # ======================
        if self.use_quant:
            if self.use_ef:
                u = delta + self.e_i
            else:
                u = delta
            u_hat = self._quantize_tensor(u, bits=self.quant_bits, group_size=self.group_size)
            residual = u - u_hat
            if self.use_ef:
                if self.ef_variant == "residual":
                    new_e = residual
                elif self.ef_variant == "momentum":
                    new_e = self.ef_beta * self.e_i + residual
                elif self.ef_variant == "normalized":
                    norm_val = torch.norm(u) + 1e-8
                    new_e = residual / norm_val
                elif self.ef_variant == "clipped":
                    new_e = torch.clamp(residual, -self.ef_clip, self.ef_clip)
                else:
                    new_e = residual
                new_e = self.ef_decay_factor * new_e
                self.e_i = new_e.detach()
            delta_hat = u_hat
            pure_quant_err = torch.norm(delta - self._quantize_tensor(delta)).item()
        else:
            delta_hat = delta
            pure_quant_err = 0.0
            if self.e_i is not None:
                self.e_i.zero_()

        # ======================
        # 更新控制变量 c_i，并可选量化上传
        # ======================
        c_i_old = self.c_i.clone() if self.use_cv else None
        if self.use_cv and c_global is not None:
            full_c_i_new = self.c_i - c_global + (w_global - w_local) / (ETA_LOCAL * actual_steps)
        else:
            full_c_i_new = None

        if self.use_cv and full_c_i_new is not None:
            delta_c = full_c_i_new - self.c_i
            # 对 delta_c 进行量化（可能使用 EF for control）
            if self.quant_control:
                ef_state = self.ef_c if self.ef_control else None
                delta_c_hat, new_ef_c, _ = self._quantize_and_feedback(
                    delta_c, ef_state, 
                    bits=self.quant_control_bits, 
                    group_size=self.quant_control_group_size,
                    use_ef=self.ef_control
                )
                if self.ef_control:
                    self.ef_c = new_ef_c.detach()
                self.c_i = self.c_i + delta_c_hat
                delta_c_for_server = delta_c_hat
            else:
                self.c_i = full_c_i_new
                delta_c_for_server = delta_c
        else:
            delta_c_for_server = None

        # 返回量化后的模型更新 delta_hat，以及控制变量更新（用于服务器聚合）
        return (delta_hat, avg_loss, torch.norm(delta).item(), pure_quant_err, delta_c_for_server)

# ======================
# Server
# ======================
class Server:
    def __init__(self, model, use_cv=True, eta_global=1.0):
        self.model = model.to(DEVICE)
        self.use_cv = use_cv
        self.eta_global = eta_global
        self.c_global = torch.zeros_like(get_flat(model)).to(DEVICE) if use_cv else None

    def aggregate(self, grads):
        return torch.stack(grads).mean(dim=0)

    def step(self, delta_mean):
        w = get_flat(self.model)
        w = w + self.eta_global * delta_mean         
        set_flat(self.model, w)

# ======================
# Test
# ======================
def test(model, loader):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            if logits.dim() == 3:
                B, T, V = logits.shape
                loss = loss_fn(logits.view(B*T, V), y.view(B*T))
                pred = logits.argmax(dim=-1)
                correct += (pred.view(-1) == y.view(-1)).sum().item()
                total += B * T
            else:
                loss = loss_fn(logits, y)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            total_loss += loss.item() * (y.size(0) if logits.dim() != 3 else B*T)
    avg_loss = total_loss / total
    return correct / total, avg_loss

# ======================
# Plotting
# ======================
def plot_all(results):
    metrics = ["train_loss", "test_acc", "grad_norm", "residual_norm",
               "quant_error", "client_drift", "compression_ratio", "optimality_gap"]
    
    # 找出所有种子的最小轮数（早停后实际轮数可能不同）
    min_rounds = float('inf')
    for name, seed_logs in results.items():
        for log in seed_logs:
            cur_rounds = len(log["round"])  # 实际记录的轮数
            if cur_rounds < min_rounds:
                min_rounds = cur_rounds
    # 如果没有任何日志，直接返回
    if min_rounds == float('inf'):
        return
    x_axis = list(range(min_rounds))

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for name, seed_logs in results.items():
            # 收集所有种子在该指标上前 min_rounds 个值
            all_curves = []
            for log in seed_logs:
                data = np.array(log[metric])[:min_rounds]  # 截取
                all_curves.append(data)
            ys = np.array(all_curves)  # shape: (num_seeds, min_rounds)
            mean_y = ys.mean(axis=0)
            std_y = ys.std(axis=0)
            plt.plot(x_axis, mean_y, label=name)
            plt.fill_between(x_axis, mean_y - std_y, mean_y + std_y, alpha=0.25)
        plt.title(f"{metric.replace('_', ' ').title()} (mean ± std)")
        plt.xlabel("Rounds")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, f"{metric}.png"))
        plt.close()
        swanlab.log({f"figures/{metric}": swanlab.Image(os.path.join(SAVE_DIR, f"{metric}.png"))})

    # 通信效率图（同样需要截取到 min_rounds）
    plt.figure(figsize=(8, 5))
    for name, seed_logs in results.items():
        all_comms = []
        all_accs = []
        for log in seed_logs:
            comms = np.array(log["comm_cost"])[:min_rounds]
            accs  = np.array(log["test_acc"])[:min_rounds]
            all_comms.append(comms)
            all_accs.append(accs)
        mean_comms = np.array(all_comms).mean(axis=0)
        mean_accs  = np.array(all_accs).mean(axis=0)
        std_accs   = np.array(all_accs).std(axis=0)
        plt.plot(mean_comms, mean_accs, label=name)
        plt.fill_between(mean_comms, mean_accs - std_accs, mean_accs + std_accs, alpha=0.25)
    plt.xlabel("Communication Cost (bits)")
    plt.ylabel("Test Accuracy")
    plt.title("Communication Efficiency (mean ± std)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "comm_efficiency.png"))
    plt.close()
    swanlab.log({"figures/comm_efficiency": swanlab.Image(os.path.join(SAVE_DIR, "comm_efficiency.png"))})

# ======================
# Single experiment runner
# ======================
def run_experiment(use_cv, use_quant, use_ef, seed,
                   ef_variant="residual", ef_beta=0.9, ef_clip=1.0, ef_decay_factor=1.0,
                   dataset="MNIST", eta_global=1.0,
                   quant_bits=8, latency=0.0, bandwidth=1e6,
                   quant_type="uniform", group_size=-1, scale_bits=16,
                   quant_control=False, ef_control=False, quant_control_bits=8,
                   quant_control_group_size=-1,
                   early_stop_patience=None, early_stop_min_delta=0.001,
                   download_quant_bits=None, download_group_size=-1, download_scale_bits=16):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global NUM_CLIENTS, ROUNDS, BATCH_SIZE, ETA_LOCAL, ETA_GLOBAL, C, NONIID, NONIID_ALPHA, FULL_PREC_BITS

    # --- 数据集加载（与原代码相同）---
    if dataset == "CIFAR10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
        clients_data = split_dataset(trainset, NUM_CLIENTS, NONIID, NONIID_ALPHA)
        clients_data = [d for d in clients_data if len(d) > 0]
        NUM_CLIENTS = len(clients_data)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        model = CNN(dataset=dataset)
        class_weights = None
    else:  # MNIST
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
        clients_data = split_dataset(trainset, NUM_CLIENTS, NONIID, NONIID_ALPHA)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        class_weights = None
        model = CNN(dataset=dataset)
        # 可选：打印客户端分布
        # labels = np.array([trainset[i][1] for i in range(len(trainset))])
        # for i, data in enumerate(clients_data):
        #     indices = data.indices
        #     client_labels = labels[indices]
        #     unique, counts = np.unique(client_labels, return_counts=True)
        #     print(f"Client {i}: total={len(data)}, class distribution: {dict(zip(unique, counts))}")

    # --- 创建服务器和客户端 ---
    server = Server(model, use_cv, eta_global=eta_global)
    clients = [Client(model=model, dataset=d,
                      use_cv=use_cv,
                      use_quant=use_quant,
                      use_ef=use_ef,
                      ef_variant=ef_variant,
                      ef_beta=ef_beta,
                      ef_clip=ef_clip,
                      quant_bits=quant_bits,
                      ef_decay_factor=ef_decay_factor,
                      class_weights=class_weights,
                      quant_type=quant_type,
                      group_size=group_size,
                      quant_control=quant_control,
                      ef_control=ef_control,
                      quant_control_bits=quant_control_bits,
                      quant_control_group_size=quant_control_group_size)
               for d in clients_data]

    # --- 通信量计算（修正版）---
    num_params = sum(p.numel() for p in server.model.parameters())
    full_prec_bits = num_params * FULL_PREC_BITS

    # 1. 上传开销（client → server）
    if use_quant and quant_type == "uniform":
        if group_size > 0:
            num_scales = (num_params + group_size - 1) // group_size
            scale_bits_total = scale_bits * num_scales
        else:
            scale_bits_total = scale_bits
        upload_model_bits = num_params * quant_bits + scale_bits_total
    elif use_quant and quant_type == "unbiased":
        upload_model_bits = num_params * quant_bits
    else:
        upload_model_bits = full_prec_bits

    if use_cv and quant_control:
        if quant_control_group_size > 0:
            num_scales_c = (num_params + quant_control_group_size - 1) // quant_control_group_size
            scale_bits_c = scale_bits * num_scales_c
        else:
            scale_bits_c = scale_bits
        upload_control_bits = num_params * quant_control_bits + scale_bits_c
    else:
        upload_control_bits = full_prec_bits if use_cv else 0

    upload_bits_per_client = upload_model_bits + upload_control_bits

    # 2. 下载开销（server → client）
    # 模型下发量化
    if download_quant_bits is not None and download_quant_bits > 0:
        if download_group_size > 0:
            num_scales_dl = (num_params + download_group_size - 1) // download_group_size
            scale_bits_dl = download_scale_bits * num_scales_dl
        else:
            scale_bits_dl = download_scale_bits
        download_model_bits = num_params * download_quant_bits + scale_bits_dl
    else:
        download_model_bits = full_prec_bits

    # 控制变量下发量化（复用 quant_control 参数，表示双向都量化）
    if use_cv:
        if quant_control:
            if quant_control_group_size > 0:
                num_scales_c_dl = (num_params + quant_control_group_size - 1) // quant_control_group_size
                scale_bits_c_dl = scale_bits * num_scales_c_dl
            else:
                scale_bits_c_dl = scale_bits
            download_control_bits = num_params * quant_control_bits + scale_bits_c_dl
        else:
            download_control_bits = full_prec_bits
    else:
        download_control_bits = 0

    download_bits_per_client = download_model_bits + download_control_bits

    # --- 早停参数初始化 ---
    if early_stop_patience is None or early_stop_patience <= 0:
        early_stop_patience = float('inf')  # 不早停

    # --- 日志初始化 ---
    logs = {
        "round": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "grad_norm": [],
        "residual_norm": [],
        "quant_error": [],
        "comm_cost": [],
        "server_c_norm": [],
        "client_drift": [],
        "compression_ratio": [],
        "optimality_gap": [],
        "comm_time": [],
    }

    cumulative_comm = 0
    cumulative_upload = 0
    cumulative_time = 0.0
    if dataset == "CIFAR10":
        target_acc = 0.65
    else:
        target_acc = 0.95
    comm_to_target = None

    # 早停状态
    best_acc = 0.0
    best_round = 0
    no_improve_rounds = 0

    # --- 联邦学习主循环 ---
    for r in range(ROUNDS):
        w_global = get_flat(server.model)

        # ===== 模拟 Server → Client 有损传输（模型量化）=====
        if download_quant_bits is not None and download_quant_bits > 0:
            q_int, scale, zero = quantize(
                w_global,
                bits=download_quant_bits,
                group_size=download_group_size,
                stochastic=True
            )
            w_global_for_client = dequantize(q_int, scale, zero)
        else:
            w_global_for_client = w_global

        # 选择客户端
        num_selected = max(1, int(C * NUM_CLIENTS))
        selected_clients = random.sample(clients, min(num_selected, NUM_CLIENTS))
        if not selected_clients:
            print(f"Round {r}: No clients selected, skipping...")
            continue

        # 通信量累计
        comm_this_round = len(selected_clients) * (upload_bits_per_client + download_bits_per_client)
        upload_this_round = len(selected_clients) * upload_bits_per_client
        cumulative_comm += comm_this_round
        cumulative_upload += upload_this_round
        cumulative_time += comm_this_round / bandwidth + latency

        losses, drifts, deltas, quant_errors, c_deltas = [], [], [], [], []
        for c in selected_clients:
            delta_hat, loss, delta_norm, quant_err, delta_c = c.train(
                w_global_for_client,   # 注意：这里传递的是量化后的模型
                server.c_global if use_cv else None
            )
            deltas.append(delta_hat)
            losses.append(loss)
            drifts.append(delta_norm)
            quant_errors.append(quant_err)
            if use_cv and delta_c is not None:
                c_deltas.append(delta_c)

        # 服务器聚合
        delta_mean = torch.stack(deltas).mean(dim=0)
        server.step(delta_mean)
        if use_cv and c_deltas:
            sampling_ratio = len(selected_clients) / NUM_CLIENTS
            delta_c_mean = torch.stack(c_deltas).mean(dim=0)
            server.c_global += sampling_ratio * delta_c_mean

        # 测试
        acc, test_loss = test(server.model, test_loader)

        # 计算指标
        avg_quant_err = sum(quant_errors) / len(quant_errors) if quant_errors else 0.0
        if use_ef:
            e_norms = [torch.norm(c.e_i) for c in clients if c.e_i is not None]
            res = torch.stack(e_norms).mean().item() if e_norms else 0.0
        else:
            res = 0.0

        # 记录日志
        logs["round"].append(r)
        logs["train_loss"].append(np.mean(losses) if losses else 0.0)
        logs["test_loss"].append(test_loss)
        logs["test_acc"].append(acc)
        logs["grad_norm"].append(torch.norm(delta_mean).item())
        logs["residual_norm"].append(res)
        logs["quant_error"].append(avg_quant_err)
        logs["client_drift"].append(np.mean(drifts) if drifts else 0.0)
        original_upload = full_prec_bits + (full_prec_bits if use_cv else 0)
        actual_upload = upload_bits_per_client
        logs["compression_ratio"].append(original_upload / actual_upload if actual_upload > 0 else 1.0)
        logs["optimality_gap"].append(1.0 - acc)
        if use_cv and server.c_global is not None:
            logs["server_c_norm"].append(torch.norm(server.c_global).item())
        else:
            logs["server_c_norm"].append(0.0)
        logs["comm_cost"].append(cumulative_comm)
        logs["comm_time"].append(cumulative_time)

        # 早停判断
        if acc > best_acc + early_stop_min_delta:
            best_acc = acc
            best_round = r
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        # 目标精度通信量记录
        if comm_to_target is None and acc >= target_acc:
            comm_to_target = cumulative_comm
            swanlab.log({"comm_efficiency_target": comm_to_target}, step=r)

        # 打印信息
        total_comm_mb = cumulative_comm / 8e6
        upload_comm_mb = cumulative_upload / 8e6
        print(f"Round {r} | Acc: {acc:.4f} | Loss: {np.mean(losses):.4f} | "
              f"TotalComm: {total_comm_mb:.2f} MB | UploadComm: {upload_comm_mb:.2f} MB")

        # swanlab 记录
        swanlab.log({
            "round": r,
            "train_loss": logs["train_loss"][-1],
            "test_acc": logs["test_acc"][-1],
            "test_loss": logs["test_loss"][-1],
            "grad_norm": logs["grad_norm"][-1],
            "residual_norm": logs["residual_norm"][-1],
            "quant_error": logs["quant_error"][-1],
            "client_drift": logs["client_drift"][-1],
            "compression_ratio": logs["compression_ratio"][-1],
            "optimality_gap": logs["optimality_gap"][-1],
            "server_c_norm": logs["server_c_norm"][-1],
            "comm_cost": logs["comm_cost"][-1],
            "upload_comm_cost": cumulative_upload,
            "comm_time": logs["comm_time"][-1],
            "total_comm_MB": cumulative_comm / 8e6,
            "upload_comm_MB": cumulative_upload / 8e6,
        }, step=r)

        # 早停退出
        if no_improve_rounds >= early_stop_patience:
            print(f"Early stop triggered at round {r}. Best acc {best_acc:.4f} at round {best_round}")
            break

    if comm_to_target is None:
        comm_to_target = cumulative_comm
        swanlab.log({"comm_efficiency_target": comm_to_target}, step=r if 'r' in locals() else 0)

    return logs

# ======================
# Main
# ======================
def main():
    global NUM_CLIENTS, ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, ETA_LOCAL, ETA_GLOBAL
    global NONIID, NONIID_ALPHA, C, SAVE_DIR

    parser = argparse.ArgumentParser(
        description="联邦学习库：量化 + Error Feedback，支持控制变量量化"
    )

    parser.add_argument("--bandwidth", type=float, default=1e6, help="网络带宽（bits/秒）")
  
    parser.add_argument("--use-quant", action="store_true", default=False, help="Enable quantization of model updates")
    parser.add_argument("--use-ef", action="store_true", default=False, help="Enable error feedback for model updates")
    parser.add_argument("--ef-decay-factor", type=float, default=1.0, help="EF衰减系数")
    parser.add_argument("--ef-variant", type=str, default="residual",
                        choices=["residual", "momentum", "normalized", "clipped"])
    parser.add_argument("--ef-beta", type=float, default=0.9)
    parser.add_argument("--ef-clip", type=float, default=1.0)
    parser.add_argument("--quant-bits", type=int, default=8, choices=[1,2,4,8,16,32])
    
    # 新增：控制变量量化相关参数
    parser.add_argument("--quant-control", action="store_true", default=False, help="Quantize control variables (c_i and c_global)")
    parser.add_argument("--ef-control", action="store_true", default=False, help="Enable error feedback for control variables")
    parser.add_argument("--quant-control-bits", type=int, default=8, choices=[1,2,4,8,16,32])
    parser.add_argument("--quant-control-group-size", type=int, default=-1, help="Group size for control variable quantization")
    
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eta-local", type=float, default=ETA_LOCAL)
    parser.add_argument("--eta-global", type=float, default=ETA_GLOBAL)
    parser.add_argument("--noniid", action="store_true", default=NONIID)
    parser.add_argument("--noniid-alpha", type=float, default=NONIID_ALPHA)
    parser.add_argument("--c", type=float, default=C)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10"])
    parser.add_argument("--quant-type", type=str, default="uniform", choices=["uniform", "unbiased"])
    parser.add_argument("--group-size", type=int, default=-1)
    parser.add_argument("--scale-bits", type=int, default=16)
    
    parser.add_argument("--download-quant-bits", type=int, default=None,
                    help="Server->Client 模型量化位数（None表示不量化）")
    parser.add_argument("--download-group-size", type=int, default=-1,
                        help="下发量化的分组大小，-1表示全局一个scale/zero")
    parser.add_argument("--download-scale-bits", type=int, default=16,
                        help="下发量化时缩放因子的量化位数（用于通信量计算）")
    
    parser.add_argument("--early-stop-patience", type=int, default=10, help="早停耐心值")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001, help="最小改善阈值")
    
    args = parser.parse_args()
    
    NUM_CLIENTS = args.num_clients
    ROUNDS = args.rounds
    LOCAL_EPOCHS = args.local_epochs
    BATCH_SIZE = args.batch_size
    ETA_LOCAL = args.eta_local
    ETA_GLOBAL = args.eta_global
    NONIID = args.noniid
    NONIID_ALPHA = args.noniid_alpha
    C = args.c
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 构造实验名称
    name = "FedAvg"    
    if args.use_quant:
        name += f"_Q{args.quant_bits}"
    if args.use_ef:
        name += "_EF"
    if args.quant_control:
        name += f"_CtrlQ{args.quant_control_bits}"
        if args.ef_control:
            name += "_CtrlEF"
    
    swanlab.init(
        project="FedJQCD_CIFAR-10_8C",
        experiment_name=name,
        config=vars(args),
        description="量化 + Error Feedback 实验，支持控制变量量化",
    )
    print(f"\n=== 运行实验配置: {name} ===")

    results = {}
    seed_logs = []
    for seed in args.seeds:
        exp_name = f"{name}_seed{seed}"
        swanlab.init(
            project="Federated_Learning",
            experiment_name=exp_name,
            config=vars(args),
            description=f"seed={seed}"
        )
        logs = run_experiment(
            True, args.use_quant, args.use_ef, seed,
            ef_variant=args.ef_variant,
            ef_beta=args.ef_beta,
            ef_clip=args.ef_clip,
            ef_decay_factor=args.ef_decay_factor,
            dataset=args.dataset,
            eta_global=ETA_GLOBAL,
            quant_bits=args.quant_bits,
            bandwidth=args.bandwidth,
            latency=args.latency,
            quant_type=args.quant_type,
            group_size=args.group_size,
            scale_bits=args.scale_bits,
            quant_control=args.quant_control,
            ef_control=args.ef_control,
            quant_control_bits=args.quant_control_bits,
            quant_control_group_size=args.quant_control_group_size,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta
        )
        seed_logs.append(logs)

    results[name] = seed_logs
    torch.save(results, os.path.join(SAVE_DIR, f"{name}_results.pt"))
    plot_all(results)
    print(f"\n实验 {name} 完成！图表已保存至 {SAVE_DIR}")

if __name__ == "__main__":
    main()