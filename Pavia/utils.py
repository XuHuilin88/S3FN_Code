from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class Mydataset(Dataset):
    def __init__(self, dataset=None, label=None, ind_img=None, transform=None):
        self.data = dataset
        self.ind_img = ind_img
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        sz = self.data.shape
        # img = torch.from_numpy(self.data[index, :, :, :].reshape(sz[1], sz[2], sz[3]))
        labels = torch.from_numpy(self.label[index])
        ids = torch.from_numpy(self.ind_img[index])
        img = np.uint8(self.data[index])
        img = np.moveaxis(img, 0, 2)
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        # m = nn.Upsample(size=32, mode='linear')
        # pos_1 = m(pos_1)
        # pos_2 = m(pos_2)

        return pos_1, pos_2, ids, labels

    def __len__(self):
        return self.data.shape[0]


class Mydataset_scan(Dataset):
    def __init__(self, dataset=None, neighbor=None, label=None, transform=None, num_neighbors=None):
        self.data = dataset
        self.neighbor = neighbor
        self.label = label
        self.transform = transform

        if num_neighbors is not None:
            self.neighbor = self.neighbor[:, :num_neighbors+1]
        assert(self.neighbor.shape[0] == len(self.data))

    def __getitem__(self, index):
        labels = torch.from_numpy(self.label[index])

        img1 = np.uint8(self.data[index])
        img1 = np.moveaxis(img1, 0, 2)
        img1 = Image.fromarray(img1)

        neighbor_index = np.random.choice(self.neighbor[index], 1)[0]
        img2 = np.uint8(self.data[neighbor_index])
        img2 = np.moveaxis(img2, 0, 2)
        img2 = Image.fromarray(img2)

        if self.transform is not None:
            pos_1 = self.transform(img1)
            pos_2 = self.transform(img2)

        # m = nn.Upsample(size=32, mode='nearest')
        # pos_1 = m(pos_1)
        # pos_2 = m(pos_2)

        return pos_1, pos_2, labels, self.neighbor[index]

    def __len__(self):
        return self.data.shape[0]


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print (path + 'Success')
        return True
    else:
        print (path + 'Exist')
        return False


def DrawResult(labels, imageID):
    # ID=1:Pavia University
    # ID=2:Indian Pines
    # ID=7:Houston
    num_class = labels.max()
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216, 191, 216],
                            [0, 255, 0],
                            [0, 255, 255],
                            [45, 138, 86],
                            [255, 0, 255],
                            [255, 165, 0],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 2:
        row = 145
        col = 145
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 2.5:
        row = 145
        col = 145
        palette = np.array([[56, 83, 163],
                            [64, 119, 188],
                            [109, 204, 220],
                            [105, 189, 69],
                            [209, 139, 188],
                            [242, 235, 23],
                            [245, 127, 33],
                            [238, 31, 35],
                            [128, 77, 77],
                            [0, 0, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 3:
        row = 349
        col = 1905
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216]])
        palette = palette * 1.0 / 255
    elif imageID == 3.5:
        row = 160
        col = 150
        palette = np.array([[0, 0, 255],
                            [0, 255, 255],
                            [0, 255, 0],
                            [255, 0, 0],
                            [255, 255, 0],
                            [128, 0, 128],
                            [255, 128, 0],
                            [0, 0, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 4:
        row = 512
        col = 217

        palette = np.array([[37, 58, 150],
                            [47, 78, 161],
                            [56, 87, 166],
                            [56, 116, 186],
                            [51, 181, 232],
                            [112, 204, 216],
                            [119, 201, 168],
                            [148, 204, 120],
                            [188, 215, 78],
                            [238, 234, 63],
                            [246, 187, 31],
                            [244, 127, 33],
                            [239, 71, 34],
                            [238, 33, 35],
                            [180, 31, 35],
                            [123, 18, 20]])
        palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0], 3))
    for i in range(1, num_class+1):
        X_result[np.where(labels == i), 0] = palette[i-1, 0]
        X_result[np.where(labels == i), 1] = palette[i-1, 1]
        X_result[np.where(labels == i), 2] = palette[i-1, 2]

    X_result = np.reshape(X_result, (row, col, 3))
    # plt.axis("off")
    # plt.imshow(X_result)
    return X_result


def CalAccuracy(predict, label):
    n = label.shape[0]
    OA = np.sum(predict == label) * 1.0 / n
    correct_sum = np.zeros((max(label) + 1))
    reali = np.zeros((max(label) + 1))
    predicti = np.zeros((max(label) + 1))
    producerA = np.zeros((max(label) + 1))

    for i in range(0, max(label) + 1):
        correct_sum[i] = np.sum(label[np.where(predict == i)] == i)
        reali[i] = np.sum(label == i)
        predicti[i] = np.sum(predict == i)
        producerA[i] = correct_sum[i] / reali[i]

    AA = np.mean(producerA)
    Kappa = (n * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (n * n - np.sum(reali * predicti))
    return AA, OA, Kappa, producerA


def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
        return X_norm
    elif type==3:
        mu = 0.5
        X_norm = X-mu
        sigma = 0.5
        X_norm = X_norm/sigma
        return X_norm



def PCANorm(X, num_PC):
    mu = np.mean(X, 0)
    X_norm = X - mu

    Sigma = np.cov(X_norm.T)
    [U, S, V] = np.linalg.svd(Sigma)
    XPCANorm = np.dot(X_norm, U[:, 0:num_PC])
    return XPCANorm


def MirrowCut(X, hw):
    # X  size: row * column * num_feature
    [row, col, n_feature] = X.shape
    # row, col,n_feature = int(row),int(col),int(n_feature)
    X_extension = np.zeros((3 * row, 3 * col, n_feature)).astype('float32')

    for i in range(0, n_feature):
        lr = np.fliplr(X[:, :, i])
        ud = np.flipud(X[:, :, i])
        lrud = np.fliplr(ud)

        l1 = np.concatenate((lrud, ud, lrud), axis=1)
        l2 = np.concatenate((lr, X[:, :, i], lr), axis=1)
        l3 = np.concatenate((lrud, ud, lrud), axis=1)
        X_extension[:, :, i] = np.concatenate((l1, l2, l3), axis=0)

    X_extension = X_extension[row - hw:2 * row + hw + 1, col - hw:2 * col + hw + 1, :]

    return X_extension


def ExtractPatches(X, w):
    if (w % 2) == 0:
        hw = int((w / 2))
        [row, col, n_feature] = X.shape
        K = row * col
        X_Mirrow = MirrowCut(X, hw)
        XP = np.zeros((K, w, w, n_feature)).astype('float32')
        for i in range(1, K + 1):
            index_row = int(np.ceil(i * 1.0 / col))
            index_col = i - (index_row - 1) * col + hw - 1
            index_row += hw - 1
            patch = X_Mirrow[index_row - hw:index_row + hw, index_col - hw:index_col + hw, :]
            XP[i - 1, :, :, :] = patch
        XP = np.moveaxis(XP, 3, 1)
        return XP
    else:
        hw = int((w/2))
        [row, col, n_feature] = X.shape
        K = row*col
        X_Mirrow = MirrowCut(X, hw)
        XP = np.zeros((K, w, w, n_feature)).astype('float32')
        for i in range(1, K+1):
            index_row = int(np.ceil(i*1.0/col))
            index_col = i - (index_row-1)*col + hw - 1
            index_row += hw - 1
            patch = X_Mirrow[index_row-hw:index_row+hw+1, index_col-hw:index_col+hw+1, :]
            XP[i-1, :, :, :] = patch
        XP = np.moveaxis(XP, 3, 1)
        return XP


def ExtractSequenPatches(X, w, array):
    if (w % 2) == 0:
        hw = int((w / 2))
        [row, col, n_feature] = X.shape
        K = array.size
        X_Mirrow = MirrowCut(X, hw)
        XP = np.zeros((K, w, w, n_feature)).astype('float32')
        X_Mirrow[hw:hw + row, hw:hw + col, :] = X
        for i in range(0, K):
            index = array[i] + 1
            index_row = int(np.ceil(index * 1.0 / col))
            index_col = index - (index_row - 1) * col + hw - 1
            index_row += hw - 1
            patch = X_Mirrow[index_row - hw:index_row + hw, index_col - hw:index_col + hw, :]
            XP[i, :, :, :] = patch
        XP = np.moveaxis(XP, 3, 1)
        return XP
    else:
        hw = int((w/2))
        [row, col, n_feature] = X.shape
        K = array.size
        X_Mirrow = MirrowCut(X, hw)
        XP = np.zeros((K, w, w, n_feature)).astype('float32')
        X_Mirrow[hw:hw+row, hw:hw+col, :] = X
        for i in range(0, K):
            index = array[i]+1
            index_row = int(np.ceil(index*1.0/col))
            index_col = index - (index_row-1)*col + hw - 1
            index_row += hw - 1
            patch = X_Mirrow[index_row-hw:index_row+hw+1, index_col-hw:index_col+hw+1, :]
            XP[i, :, :, :] = patch
        XP = np.moveaxis(XP, 3, 1)
        return XP


def SamplesGenerate_all(X, Y, train_num_array, w, n_class):
    [row, col, n_feature] = X.shape
    Y = Y.reshape(row * col, 1)
    train_num_all = sum(train_num_array)
    X_train_test = np.zeros((train_num_all, n_feature, w, w)).astype('float32')
    Y_train_test = np.zeros((train_num_all, 1)).astype('float32')
    index_train_test = np.zeros((train_num_all, 1)).astype('int')

    flag1 = 0
    new_label = 1
    for i in range(1, n_class + 1):
        index = np.where(Y == i)[0]
        # np.random.seed(0)
        if train_num_array[i - 1] != 0:
            tem = ExtractSequenPatches(X, w, index[0:train_num_array[i - 1]])
            X_train_test[flag1:flag1 + train_num_array[i - 1], :, :, :] = tem
            Y[index, 0] = new_label
            Y_train_test[flag1:flag1 + train_num_array[i - 1], 0] = Y[index, 0]
            index_train_test[flag1:flag1 + train_num_array[i - 1], 0] = index
            new_label = new_label + 1

        flag1 = flag1 + train_num_array[i - 1]

    X_all = ExtractPatches(X, w)

    return X_all, X_train_test, Y_train_test, index_train_test


def SamplesGenerate(X, Y, train_num_array, w, n_class):
    [row, col, n_feature] = X.shape
    Y = Y.reshape(row * col, 1)
    train_num_all = sum(train_num_array)
    X_train = np.zeros((train_num_all, n_feature, w, w)).astype('float32')
    Y_train = np.zeros((train_num_all, 1)).astype('float32')
    index_train = np.zeros((train_num_all, 1)).astype('int')

    X_test = np.zeros((sum(Y > 0)[0] - train_num_all, n_feature, w, w)).astype('float32')
    Y_test = np.zeros((sum(Y > 0)[0] - train_num_all, 1)).astype('float32')
    index_test = np.zeros((sum(Y > 0)[0] - train_num_all, 1)).astype('int')

    flag1 = 0
    flag2 = 0
    for i in range(1, n_class + 1):
        index = np.where(Y == i)[0]
        n_data = index.shape[0]
        # np.random.seed(0)
        randomArray = (np.random.permutation(n_data))
        tem = ExtractSequenPatches(X, w, index[randomArray[0:train_num_array[i - 1]]])
        X_train[flag1:flag1 + train_num_array[i - 1], :, :, :] = tem
        Y_train[flag1:flag1 + train_num_array[i - 1], 0] = Y[index[randomArray[0:train_num_array[i - 1]]], 0]
        index_train[flag1:flag1 + train_num_array[i - 1], 0] = index[randomArray[0:train_num_array[i - 1]]]

        tem = ExtractSequenPatches(X, w, index[randomArray[train_num_array[i - 1]:n_data]])
        X_test[flag2:flag2 + n_data - train_num_array[i - 1], :, :, :] = tem
        Y_test[flag2:flag2 + n_data - train_num_array[i - 1], 0] = Y[index[randomArray[train_num_array[i - 1]:n_data]], 0]
        index_test[flag2:flag2 + n_data - train_num_array[i - 1], 0] = index[randomArray[train_num_array[i - 1]:n_data]]

        flag1 = flag1 + train_num_array[i - 1]
        flag2 = flag2 + n_data - train_num_array[i - 1]

    X_all = ExtractPatches(X, w)
    return X_all, X_train, Y_train, index_train, X_test, Y_test, index_test


def SamplesGenerate_spe(X_ori, X, Y, train_num_array, w, n_class):
    [row, col, n_feature] = X.shape
    [_, _, n_dim] = X_ori.shape
    X_ori = X_ori.reshape(row * col, n_dim)
    Y = Y.reshape(row * col, 1)
    train_num_all = sum(train_num_array)
    X_train = np.zeros((train_num_all, n_feature, w, w)).astype('float32')
    Pixel_train = np.zeros((train_num_all, n_dim)).astype('float32')
    Y_train = np.zeros((train_num_all, 1)).astype('float32')
    index_train = np.zeros((train_num_all, 1)).astype('int')

    X_test = np.zeros((sum(Y > 0)[0] - train_num_all, n_feature, w, w)).astype('float32')
    Pixel_test = np.zeros((sum(Y > 0)[0] - train_num_all, n_dim)).astype('float32')
    Y_test = np.zeros((sum(Y > 0)[0] - train_num_all, 1)).astype('float32')
    index_test = np.zeros((sum(Y > 0)[0] - train_num_all, 1)).astype('int')

    flag1 = 0
    flag2 = 0
    for i in range(1, n_class + 1):
        index = np.where(Y == i)[0]
        n_data = index.shape[0]
        # np.random.seed(0)
        randomArray = (np.random.permutation(n_data))
        tem = ExtractSequenPatches(X, w, index[randomArray[0:train_num_array[i - 1]]])
        X_train[flag1:flag1 + train_num_array[i - 1], :, :, :] = tem
        Pixel_train[flag1:flag1 + train_num_array[i - 1], :] = X_ori[index[randomArray[0:train_num_array[i - 1]]], :]
        Y_train[flag1:flag1 + train_num_array[i - 1], 0] = Y[index[randomArray[0:train_num_array[i - 1]]], 0]
        index_train[flag1:flag1 + train_num_array[i - 1], 0] = index[randomArray[0:train_num_array[i - 1]]]

        tem = ExtractSequenPatches(X, w, index[randomArray[train_num_array[i - 1]:n_data]])
        X_test[flag2:flag2 + n_data - train_num_array[i - 1], :, :, :] = tem
        Pixel_test[flag2:flag2 + n_data - train_num_array[i - 1], :] = X_ori[index[randomArray[train_num_array[i - 1]:n_data]], :]
        Y_test[flag2:flag2 + n_data - train_num_array[i - 1], 0] = Y[index[randomArray[train_num_array[i - 1]:n_data]], 0]
        index_test[flag2:flag2 + n_data - train_num_array[i - 1], 0] = index[randomArray[train_num_array[i - 1]:n_data]]

        flag1 = flag1 + train_num_array[i - 1]
        flag2 = flag2 + n_data - train_num_array[i - 1]

    X_all = ExtractPatches(X, w)
    return X_all, X_train, Pixel_train, Y_train, index_train, X_test, Pixel_test, Y_test, index_test


def Index_SamplesGenerate(X_ori, X, Y, train_num_array, index_train, index_test, w):
    [row, col, n_pc] = X.shape
    Y = Y.reshape(row * col, 1)
    index_all = np.arange(row*col).reshape(-1, 1)
    train_num_all = sum(train_num_array)
    X_train = np.zeros((train_num_all, n_pc, w, w)).astype('float32')
    Y_train = np.zeros((train_num_all, 1)).astype('float32')

    X_test = np.zeros((sum(Y > 0)[0] - train_num_all, n_pc, w, w)).astype('float32')
    Y_test = np.zeros((sum(Y > 0)[0] - train_num_all, 1)).astype('float32')

    # np.random.seed(0)
    tem = ExtractSequenPatches(X, w, index_train[:, 0])
    X_train[:, :, :, :] = tem
    Y_train[:, 0] = Y[index_train[:, 0], 0]

    tem = ExtractSequenPatches(X, w, index_test[:, 0])
    X_test[:, :, :, :] = tem
    Y_test[:, 0] = Y[index_test[:, 0], 0]

    X_all = ExtractPatches(X, w)

    return X_all, index_all, X_train, Y_train, X_test, Y_test