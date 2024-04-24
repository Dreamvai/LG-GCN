'''Predicting'''
from scipy.io import loadmat, savemat
import numpy as np
import argparse
import configparser
import torch
from torch_geometric.data import Data, Batch
from skimage.segmentation import slic, mark_boundaries
from sklearn.preprocessing import scale
import os
from PIL import Image
from utils import get_graph_list, get_edge_index
import math
from train import Trainer, Wangyufan, applyPCA

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN THE OVERALL')
    parser.add_argument('--name', type=str, default='Indian_pines',help='DATASET NAME')
    parser.add_argument('--comp', type=int, default=10, help='COMPACTNESS')
    parser.add_argument('--batchsz', type=int, default=64, help='BATCH SIZE')
    parser.add_argument('--run', type=int, default=10, help='EXPERIMENT AMOUNT')
    parser.add_argument('--spc', type=int, default=30, help='SAMPLE per CLASS')
    parser.add_argument('--hsz', type=int, default=128, help='HIDDEN SIZE')
    parser.add_argument('--lr', type=float, default=1e-3, help='LEARNING RATE')
    parser.add_argument('--wd', type=float, default=0., help='WEIGHT DECAY')
    parser.add_argument('--pca', type=float, default=103, help='PCA')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')
    if arg.name == 'PaviaU':
        block = [5, 10, 15, 20, 25, 30, 35, 40]
    if arg.name == 'Salinas':
        block = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    if arg.name == 'Indian_pines':
        block = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    print(block)
    for block in block:
        print('这是block=', block)
        # Data processing
        # Reading hyperspectral image
        if arg.name == 'Indian_pines':
            data_path = '{0}/{0}_corrected.mat'.format(arg.name)
        else:
            data_path = '{0}/{0}.mat'.format(arg.name)
        # data_path = '{0}/{0}.mat'.format(arg.name)
        m = loadmat(data_path)
        data = m[config.get(arg.name, 'data_key')]
        #data = applyPCA(data, numComponents=arg.pca)
        gt_path = '{0}/{0}_gt.mat'.format(arg.name)
        m = loadmat(gt_path)
        gt = m[config.get(arg.name, 'gt_key')]
        # Normalizing data
        h, w, c = data.shape
        data = data.reshape((h * w, c))
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

        gcn1 = Wangyufan(c, arg.hsz, config.getint(arg.name, 'nc'))
        # gcn1 = SubGcnFeature(config.getint(arg.name, 'band'), arg.hsz)
        # gcn2 = GraphNet(arg.hsz, arg.hsz, config.getint(arg.name, 'nc'))

        # device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for r in range(arg.run):
            # Loading pretraining parameters
            gcn1.load_state_dict(
                torch.load(
                    f"models/{arg.name}/{arg.spc}/PCA{c}/{block}_overall_skip_2_SGConv_l1_clip/intNet_best_{arg.spc}_{r}.pkl"))
            trainer = Trainer(gcn1)
            # predicting
            preds = trainer.predict(subGraph, fullGraph, device)
            seg_torch = torch.from_numpy(seg)
            map = preds[seg_torch]
            save_root = 'prediction/{}/{}/PCA{}/{}_overall_skip_2_SGConv_l1_clip'.format(arg.name, arg.spc, c, block)
            # print(save_root)
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_path = os.path.join(save_root, '{}.mat'.format(r))
            savemat(save_path, {'pred': map.cpu().numpy()})
        print('*' * 5 + 'FINISH' + '*' * 5)


