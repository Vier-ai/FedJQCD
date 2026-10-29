import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        # Analytical payload size, matching standalone FedJQCD: FP32 parameters,
        # no transport headers, no evaluation traffic, decimal MB = bits / 8e6.
        self.model_bits = sum(p.numel() for p in self.global_model.parameters()) * 32
        self.cumulative_upload = 0
        self.cumulative_comm = 0
        self.cumulative_comm_time = 0.0
        self.comm_to_target = None
        self.round_history = []
        self.swanlab = None
        if args.swanlab_mode != "disabled":
            import swanlab
            self.swanlab = swanlab

    @staticmethod
    def parameter_vector(model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def begin_round(self):
        self.round_start_vector = self.parameter_vector(self.global_model).clone()
        self.round_download_bits = 0

    def record_round_updates(self):
        # Capture deltas before aggregation mutates the server model.
        delta_sum = torch.zeros_like(self.round_start_vector)
        drifts = []
        for cid in self.uploaded_ids:
            delta = self.parameter_vector(self.clients[cid].model) - self.round_start_vector
            delta_sum += delta
            drifts.append(delta.norm().item())
        self.round_grad_norm = (delta_sum / len(self.uploaded_ids)).norm().item()
        self.round_client_drift = float(np.mean(drifts))
        self.round_train_loss = float(np.mean(
            [c.last_train_loss for c in self.selected_clients]))
        multiplier = 2 if self.algorithm == "SCAFFOLD" else 1
        upload = len(self.uploaded_ids) * self.model_bits * multiplier
        total = upload + self.round_download_bits
        self.cumulative_upload += upload
        self.cumulative_comm += total
        self.cumulative_comm_time += total / self.args.bandwidth + self.args.latency

    def log_round(self, round_index):
        metrics = {
            "round": round_index,
            "train_loss": self.round_train_loss,
            "grad_norm": self.round_grad_norm,
            "residual_norm": 0.0,
            "quant_error": 0.0,
            "client_drift": self.round_client_drift,
            "compression_ratio": 1.0,
            "server_c_norm": (torch.cat([c.detach().reshape(-1) for c in self.global_c]).norm().item()
                              if self.algorithm == "SCAFFOLD" else 0.0),
            "comm_cost": self.cumulative_comm,
            "upload_comm_cost": self.cumulative_upload,
            "comm_time": self.cumulative_comm_time,
            "total_comm_MB": self.cumulative_comm / 8e6,
            "upload_comm_MB": self.cumulative_upload / 8e6,
        }
        if round_index % self.eval_gap == 0 or round_index == self.global_rounds - 1:
            self.evaluate()
            metrics.update(test_acc=self.rs_test_acc[-1], test_loss=self.last_test_loss,
                           optimality_gap=1.0 - self.rs_test_acc[-1])
            if (self.args.target_acc is not None and self.comm_to_target is None
                    and metrics["test_acc"] >= self.args.target_acc):
                self.comm_to_target = self.cumulative_comm
                metrics["comm_efficiency_target"] = self.comm_to_target
                metrics["target_reached"] = 1
        self.round_history.append(metrics)
        if self.swanlab is not None:
            self.swanlab.log(metrics, step=round_index)
        print(f"Round {round_index} | TotalComm: {metrics['total_comm_MB']:.2f} MB | "
              f"UploadComm: {metrics['upload_comm_MB']:.2f} MB")

    def finish_communication_log(self):
        if self.comm_to_target is None and self.args.target_acc is not None:
            # Keep the standalone field, explicitly mark an unreached target.
            metrics = {"comm_efficiency_target": self.cumulative_comm, "target_reached": 0}
            self.round_history[-1].update(metrics)
            if self.swanlab is not None:
                self.swanlab.log(metrics, step=self.round_history[-1]["round"])

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)
            self.round_download_bits += self.model_bits

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                history = hf.create_group('round_metrics')
                for key in sorted(set().union(*(row.keys() for row in self.round_history))):
                    history.create_dataset(key, data=[row.get(key, np.nan) for row in self.round_history])

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # Evaluate the aggregated global model, after this round's communication.
    def evaluate(self, acc=None, loss=None):
        clients = self.clients
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            clients = self.new_clients
        correct, samples, auc_sum, test_loss_sum = 0, 0, 0.0, 0.0
        for client in clients:
            local_model = client.model
            if not self.eval_new_clients:
                client.model = self.global_model
            try:
                ct, ns, auc = client.test_metrics()
                correct += ct
                samples += ns
                auc_sum += auc * ns
                test_loss_sum += client.last_test_loss_sum
            finally:
                client.model = local_model
        test_acc = correct / samples
        self.last_test_loss = test_loss_sum / samples
        (self.rs_test_acc if acc is None else acc).append(test_acc)
        (self.rs_train_loss if loss is None else loss).append(self.round_train_loss)
        self.rs_test_auc.append(auc_sum / samples)
        print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {self.last_test_loss:.4f}")

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
