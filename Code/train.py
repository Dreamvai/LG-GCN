'''Training'''
###python -m visdom.server  在终端先运行这行code
from scipy.io import loadmat
import numpy as np
from thop import profile
from sklearn.decomposition import PCA
from torch.nn import functional as F
import argparse
import configparser
from torch import nn
from skimage.segmentation import slic
from sklearn.preprocessing import scale, minmax_scale
import os
from PIL import Image
from utils import get_graph_list, split, get_edge_index
import math
from Monitor import GradMonitor
from visdom import Visdom
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import accuracy
from torch.nn.utils import clip_grad_norm_
import time
from torch_geometric import nn as gnn
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

##模型
class Wangyufan(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=1)
        self.gcnx = gnn.SGConv(hidden_size, hidden_size+28, K=1)
        self.gcny = gnn.SGConv(hidden_size+28, hidden_size + 56, K=1)

        self.bn_0 = gnn.BatchNorm(hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size + 28)
        self.bn_2 = gnn.BatchNorm(hidden_size + 56)
        self.bn_3 = gnn.BatchNorm(hidden_size+84)

        self.gcn_1 = gnn.SGConv(hidden_size, hidden_size+28, K=2)
        self.gcn_2 = gnn.SGConv(hidden_size+28, hidden_size+56, K=2)
        self.gcn_3 = gnn.SGConv(hidden_size + 56, hidden_size + 84, K=2)

        # self.gcn_3 = gnn.GraphConv(hidden_size, hidden_size)
        # self.bn_3 = gnn.BatchNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size+84, (hidden_size+84) // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear((hidden_size+84) // 2, nc)
        )
    def forward(self, graph, fullGraph):
        h1 = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg1 = gnn.global_mean_pool(h1, graph.batch)
        fullGraph.x = h_avg1
        x_normalization1 = self.bn_0(fullGraph.x)

        h2 = F.relu(self.gcnx(h1, graph.edge_index))
        h_avg2 = gnn.global_mean_pool(h2, graph.batch)
        x_normalization2 = self.bn_1(h_avg2)

        h3 = F.relu(self.gcny(h2, graph.edge_index))
        h_avg3 = gnn.global_mean_pool(h3, graph.batch)
        x_normalization3 = self.bn_2(h_avg3)


        h = self.bn_1(F.relu(self.gcn_1(x_normalization1, fullGraph.edge_index)))
        h = h + x_normalization2
        h = self.bn_2(F.relu(self.gcn_2(h, fullGraph.edge_index)))
        h = h + x_normalization3
        h = self.bn_3(F.relu(self.gcn_3(h, fullGraph.edge_index)))

        logits = self.classifier(h)

        return logits

class Trainer(object):
    r'''Joint trainer'''
    def __init__(self, models: list):
        super().__init__()
        self.models = models

    def train(self, subGraph: Batch, fullGraph: Data, optimizer, criterion, device, monitor = None, is_l1=False, is_clip=False):
        intNet = self.models
        intNet.train()
        intNet.to(device)
        criterion.to(device)
        subGraph = subGraph.to(device)
        fullGraph = fullGraph.to(device)
        logits = intNet(subGraph, fullGraph)
        indices = torch.nonzero(fullGraph.tr_gt, as_tuple=True)
        y = fullGraph.tr_gt[indices].to(device) - 1
        node_number = fullGraph.seg[indices]
        pixel_logits = logits[node_number]
        loss = criterion(pixel_logits, y)
        # l1 norm
        if is_l1:
            l1 = 0
            for p in intNet.parameters():
                l1 += p.norm(1)
            loss += 1e-4 * l1
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        # Clipping gradient
        if is_clip:
            clip_grad_norm_(intNet.parameters(), max_norm=2., norm_type=2)
        optimizer.step()

        if monitor is not None:
            monitor.add([intNet.parameters()], ord=2)
        return loss.item()

    def evaluate(self, subGraph, fullGraph, criterion, device):
        intNet = self.models
        intNet.train()
        intNet.to(device)
        criterion.to(device)
        with torch.no_grad():
            subGraph = subGraph.to(device)
            fullGraph = fullGraph.to(device)
            logits = intNet(subGraph, fullGraph)
            pred = torch.argmax(logits, dim=-1)
            indices = torch.nonzero(fullGraph.te_gt, as_tuple=True)
            y = fullGraph.te_gt[indices].to(device) - 1
            node_number = fullGraph.seg[indices]
            pixel_pred = pred[node_number]
            pixel_logits = logits[node_number]
            loss = criterion(pixel_logits, y)
        return loss.item(), accuracy(pixel_pred, y)

    # Getting prediction results
    def predict(self, subGraph, fullGraph, device: torch.device):
        intNet = self.models
        intNet.train()
        intNet.to(device)
        with torch.no_grad():
            subGraph = subGraph.to(device)
            fullGraph = fullGraph.to(device)
            logits = intNet(subGraph, fullGraph)
        pred = torch.argmax(logits, dim=-1)
        return pred

    def get_parameters(self):
        return self.models

    def save(self, paths):
        torch.save(self.models.cpu().state_dict(), paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN SUBGRAPH')
    parser.add_argument('--name', type=str, default='Indian_pines', help='DATASET NAME')
    parser.add_argument('--epoch', type=int, default=500, help='ITERATION')
    parser.add_argument('--comp', type=int, default=10, help='COMPACTNESS')
    parser.add_argument('--batchsz', type=int, default=64, help='BATCH SIZE')
    parser.add_argument('--run', type=int, default=10, help='EXPERIMENT AMOUNT')
    parser.add_argument('--spc', type=int, default=30, help='SAMPLE per CLASS')
    parser.add_argument('--hsz', type=int, default=128, help='HIDDEN SIZE')
    parser.add_argument('--lr', type=float, default=1e-3, help='LEARNING RATE')
    parser.add_argument('--wd', type=float, default=0., help='WEIGHT DECAY')
    parser.add_argument('--pca', type=float, default=50, help='PCA')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')
    viz = Visdom(port=8097)
    if arg.name == 'PaviaU':
        block = [5, 10, 15, 20, 25, 30, 35, 40]
    if arg.name == 'Salinas':
        block = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    if arg.name == 'Indian_pines':
        block = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    print(block)
    block = [5]
    for block in block:
        train_time = 0
        Total_time = 0
        print('这是block=', block)
        # Data processing
        # Reading hyperspectral image
        if arg.name == 'Indian_pines':
            data_path = '{0}/{0}_corrected.mat'.format(arg.name)
        else:
            data_path = '{0}/{0}.mat'.format(arg.name)
        m = loadmat(data_path)
        data = m[config.get(arg.name, 'data_key')]
        #data = applyPCA(data, numComponents=arg.pca)
        gt_path = '{0}/{0}_gt.mat'.format(arg.name)
        m = loadmat(gt_path)
        gt = m[config.get(arg.name, 'gt_key')]
        # Normalizing data
        h, w, c = data.shape
        print(c)
        data = data.reshape((h * w, c))
        data = data / 1.0
        if arg.name == 'Xiongan':
            minmax_scale(data, copy=False)
        data_normalization = scale(data).reshape((h, w, c))

        # Superpixel segmentation
        seg_root = 'rgb'
        seg_path = os.path.join(seg_root, '{}_seg_{}.npy'.format(arg.name, block))
        if os.path.exists(seg_path):
            seg = np.load(seg_path)
        else:
            rgb_path = os.path.join(seg_root, '{}_rgb.jpg'.format(arg.name))
            img = Image.open(rgb_path)
            img_array = np.array(img)
            # The number of superpixel
            n_superpixel = int(math.ceil((h * w) / block))
            seg = slic(img_array, n_superpixel, arg.comp)

            # Saving
            np.save(seg_path, seg)

        # Constructing graphs
        graph_path = '{}/PCA{}/{}_graph.pkl'.format(arg.name, c, block)
        if os.path.exists(graph_path):
            graph_list = torch.load(graph_path)
        else:
            graph_list = get_graph_list(data_normalization, seg)
            torch.save(graph_list, graph_path)
        subGraph = Batch.from_data_list(graph_list)

        # Constructing full graphs
        full_edge_index_path = '{}/PCA{}/{}_edge_index.npy'.format(arg.name, c, block)
        if os.path.exists(full_edge_index_path):
            edge_index = np.load(full_edge_index_path)
        else:
            edge_index, _ = get_edge_index(seg)
            np.save(full_edge_index_path,
                    edge_index if isinstance(edge_index, np.ndarray) else edge_index.cpu().numpy())
        fullGraph = Data(None,
                         edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                         seg=torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg)

        for r in range(arg.run):
            print('\n')
            print('*' * 5 + 'Run {}'.format(r) + '*' * 5)
            # Reading the training data set and testing data set
            m = loadmat('trainTestSplit/{}/sample{}_run{}.mat'.format(arg.name, arg.spc, r))
            tr_gt, te_gt = m['train_gt'], m['test_gt']
            tr_gt_torch, te_gt_torch = torch.from_numpy(tr_gt).long(), torch.from_numpy(te_gt).long()
            fullGraph.tr_gt, fullGraph.te_gt = tr_gt_torch, te_gt_torch
            gcn1 = Wangyufan(c, arg.hsz, config.getint(arg.name, 'nc'))
            #gcn1.apply(init_weights)
            optimizer = torch.optim.Adam([{'params': gcn1.parameters()}], lr=arg.lr, weight_decay=arg.wd)
            criterion = nn.CrossEntropyLoss()
            trainer = Trainer(gcn1)
            monitor = GradMonitor()

            # Plotting a learning curve and gradient curve
            viz.line([[0., 0., 0.]], [0], win='{}_train_test_acc_{}'.format(arg.name, r),
                     opts={'title': '{} train&test&acc {}'.format(arg.name, r),
                           'legend': ['train', 'test', 'acc']})
            viz.line([[0., 0.]], [0], win='{}_grad_{}'.format(arg.name, r),
                     opts={'title': '{} grad {}'.format(arg.name, r),
                           'legend': ['internal', 'external']})

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')
            max_acc = 0
            save_root = 'models/{}/{}/PCA{}/{}_overall_skip_2_SGConv_l1_clip'.format(arg.name, arg.spc, c, block)
            pbar = tqdm(range(arg.epoch))
            time_start2 = time.time()  # 记录开始时间
            # Training
            for epoch in pbar:
                pbar.set_description_str('Epoch: {}'.format(epoch))
                time_start1 = time.time()  # 记录开始时间
                tr_loss = trainer.train(subGraph, fullGraph, optimizer, criterion, device, monitor.clear(), is_l1=True, is_clip=True)
                time_end1 = time.time()
                time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
                train_time = train_time + time_sum1  #记录耗时
                te_loss, acc = trainer.evaluate(subGraph, fullGraph, criterion, device)
                pbar.set_postfix_str('train loss: {} test loss:{} acc:{}'.format(tr_loss, te_loss, acc))
                viz.line([[tr_loss, te_loss, acc]], [epoch], win='{}_train_test_acc_{}'.format(arg.name, r), update='append')
                viz.line([monitor.get()], [epoch], win='{}_grad_{}'.format(arg.name, r), update='append')

                if acc > max_acc:
                    max_acc = acc
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    #trainer.save(os.path.join(save_root, 'intNet_best_{}_{}.pkl'.format(arg.spc, r)))  ##保存模型
            time_end2 = time.time()
            Total_time = time_end2 - time_start2  # 计算的时间差为程序的执行时间，单位为秒/s
            print(f'第{r}次的训练时间:', train_time)
            print(f'第{r}次的训练+测试的总时间:', Total_time)
        print('*' * 5 + 'FINISH' + '*' * 5)











