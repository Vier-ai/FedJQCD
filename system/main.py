#!/usr/bin/env python
import argparse
import os
import random
import time
from functools import partial

import numpy as np
import torch
from torch import nn
import torchvision

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverdyn import FedDyn
from flcore.trainmodel.models import JQCDCNN, LSTMNet
from utils.result_utils import average_data


def build_model(args):
    if args.model == "CNN":
        return JQCDCNN(args.dataset).to(args.device)
    if args.model == "ResNet18":
        if args.dataset == "Shakespeare":
            raise ValueError("Shakespeare uses LSTM; ResNet18 is an image model.")
        model = torchvision.models.resnet18(
            weights=None, num_classes=args.num_classes,
            norm_layer=partial(nn.GroupNorm, 32))
        model.conv1 = nn.Conv2d(
            1 if args.dataset == "MNIST" else 3, 64,
            kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model.to(args.device)
    return LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size,
                   num_classes=args.num_classes).to(args.device)


def run(args):
    servers = {"FedAvg": FedAvg, "FedProx": FedProx,
               "SCAFFOLD": SCAFFOLD, "FedDyn": FedDyn}
    model_name = args.model
    config = vars(args).copy()
    time_list = []
    for i in range(args.prev, args.times):
        seed = args.seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.model = model_name
        args.model = build_model(args)
        print(f"\n============= {args.algorithm}: seed {seed} =============")
        start = time.time()
        swanlab = None
        if args.swanlab_mode != "disabled":
            import swanlab
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=f"{args.algorithm}_{args.dataset}_{model_name}_seed{seed}",
                config={**config, "seed": seed, "comm_unit": "bit",
                        "MB_divisor": 8e6, "full_precision_bits": 32,
                        "resnet_norm": "GroupNorm" if model_name == "ResNet18" else None,
                        "evaluation": "pooled_client_test_samples",
                        "aggregation": "sample_weighted" if args.algorithm in ("FedAvg", "FedProx") else "algorithm_default"},
                mode=args.swanlab_mode)
        try:
            server = servers[args.algorithm](args, i)
            server.train()
        finally:
            if swanlab is not None:
                swanlab.finish()
            args.model = model_name
        time_list.append(time.time() - start)
    print(f"\nAverage time cost: {np.mean(time_list):.2f}s.")
    if args.prev == 0:
        average_data(dataset=args.dataset, algorithm=args.algorithm,
                     goal=args.goal, times=args.times)


def parse_args():
    parser = argparse.ArgumentParser(description="FedJQCD comparison baselines")
    parser.add_argument("-algo", "--algorithm", choices=["FedAvg", "FedProx", "SCAFFOLD", "FedDyn"], default="FedAvg")
    parser.add_argument("-data", "--dataset", choices=["MNIST", "Cifar10", "CIFAR10", "Shakespeare"], default="MNIST")
    parser.add_argument("-m", "--model", choices=["CNN", "ResNet18", "LSTM"], default=None)
    parser.add_argument("-ncl", "--num_classes", type=int, default=None)
    parser.add_argument("-dev", "--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("-did", "--device_id", default="0")
    parser.add_argument("-nc", "--num_clients", type=int, default=20)
    parser.add_argument("-jr", "--join_ratio", type=float, default=1.0)
    parser.add_argument("-rjr", "--random_join_ratio", action="store_true")
    parser.add_argument("-gr", "--global_rounds", type=int, default=2000)
    parser.add_argument("-ls", "--local_epochs", type=int, default=1)
    parser.add_argument("-lbs", "--batch_size", type=int, default=10)
    parser.add_argument("-lr", "--local_learning_rate", type=float, default=0.005)
    parser.add_argument("-ld", "--learning_rate_decay", action="store_true")
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument("-eg", "--eval_gap", type=int, default=1)
    parser.add_argument("-t", "--times", type=int, default=1)
    parser.add_argument("-pv", "--prev", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-go", "--goal", default="test")
    parser.add_argument("-sfn", "--save_folder_name", default="items")
    parser.add_argument("-ab", "--auto_break", action="store_true")
    parser.add_argument("-tc", "--top_cnt", type=int, default=100)
    parser.add_argument("-mu", "--mu", type=float, default=0.0, help="FedProx coefficient")
    parser.add_argument("-al", "--alpha", type=float, default=1.0, help="FedDyn coefficient")
    parser.add_argument("-slr", "--server_learning_rate", type=float, default=1.0)
    parser.add_argument("-fd", "--feature_dim", type=int, default=512)
    parser.add_argument("-vs", "--vocab_size", type=int, default=80)
    parser.add_argument("-fs", "--few_shot", type=int, default=0)
    parser.add_argument("-cdr", "--client_drop_rate", type=float, default=0.0)
    parser.add_argument("-tsr", "--train_slow_rate", type=float, default=0.0)
    parser.add_argument("-ssr", "--send_slow_rate", type=float, default=0.0)
    parser.add_argument("-ts", "--time_select", action="store_true")
    parser.add_argument("-tth", "--time_threthold", type=float, default=10000)
    parser.add_argument("-dlg", "--dlg_eval", action="store_true")
    parser.add_argument("-dlgg", "--dlg_gap", type=int, default=100)
    parser.add_argument("-bnpc", "--batch_num_per_client", type=int, default=2)
    parser.add_argument("-nnc", "--num_new_clients", type=int, default=0)
    parser.add_argument("-ften", "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument("--bandwidth", type=float, default=1e6, help="bits/second, same simulation as FedJQCD")
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--target_acc", type=float, default=None)
    parser.add_argument("--swanlab_project", default="Federated_Learning")
    parser.add_argument("--swanlab_mode", choices=["online", "local", "offline", "disabled"], default="online")
    args = parser.parse_args()
    if args.dataset == "CIFAR10":
        args.dataset = "Cifar10"
    if args.model is None:
        args.model = "LSTM" if args.dataset == "Shakespeare" else "CNN"
    if (args.dataset == "Shakespeare") != (args.model == "LSTM"):
        parser.error("Shakespeare requires LSTM; image datasets require CNN or ResNet18.")
    expected_classes = 80 if args.dataset == "Shakespeare" else 10
    if args.num_classes is None:
        args.num_classes = expected_classes
    if args.num_classes != expected_classes:
        parser.error(f"{args.dataset} requires num_classes={expected_classes}.")
    if args.target_acc is None:
        args.target_acc = 0.65 if args.dataset == "Cifar10" else 0.95 if args.dataset == "MNIST" else None
    if args.global_rounds < 1 or args.local_epochs < 1 or args.eval_gap < 1:
        parser.error("global_rounds, local_epochs and eval_gap must be positive.")
    if not 0 <= args.prev < args.times:
        parser.error("Require 0 <= prev < times.")
    if not 0 < args.join_ratio <= 1 or int(args.num_clients * args.join_ratio) < 1:
        parser.error("join_ratio must select at least one client.")
    if args.bandwidth <= 0 or args.latency < 0:
        parser.error("bandwidth must be positive and latency must be nonnegative.")
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    run(args)
