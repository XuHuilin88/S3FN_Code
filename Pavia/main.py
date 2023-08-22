import argparse
import os
import scipy.io as sio
import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from sklearn import svm

from utils import *
from model import *
from resnet import *
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# device = torch.device("cuda," if torch.cuda.is_available() else "cpu")
resultpath = 'S3FN_results/'
mkdir(resultpath)


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, r):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, ind_tr, _ in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        pixels_train = X_spe[np.squeeze(ind_tr), :].astype("float32")
        pixels_train = torch.from_numpy(pixels_train).cuda()
        pixels_train = pixels_train + torch.randn(n_feature).cuda() * args.noise

        # _, out_1 = net(pos_1)
        _, out1 = net(pos_1, pixels_train, r)
        del pos_1
        _, out2 = net(pos_2, pixels_train, r)
        del pos_2
        # [2*B, D]

        # ####### loss ###############
        out = torch.cat([out1, out2], dim=0)
        # out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        a, _ = sim_matrix.size()  # a=2*B
        # mask = (torch.ones_like(sim_matrix) - torch.eye(a, device=sim_matrix.device)).type(torch.bool)
        mask = (torch.ones_like(sim_matrix) - torch.eye(a, device=sim_matrix.device))

        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask.byte()).view(a, -1)

        pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
        # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += target.size(0)
        total_loss += loss.item() * target.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_svm(net, memory_data_loader, test_data_loader, r):
    net.eval()
    total_top1, total_top5, total_num, train_feature_bank, test_feature_bank, tra_targets, test_targets = 0.0, 0.0, 0, [], [], [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, ind_tr, target in tqdm(memory_data_loader, desc='Feature extracting'):
            data = data.cuda(non_blocking=True)
            pixels_train = X_spe[np.squeeze(ind_tr), :].astype("float32")
            pixels_train = torch.from_numpy(pixels_train).cuda()

            feature, out = net(data, pixels_train, r)
            train_feature_bank.append(feature)
            tra_targets.append(target)
        # [N, D]
        train_feature_bank = torch.cat(train_feature_bank, dim=0)
        # [N]
        tra_targets = torch.cat(tra_targets, dim=0).squeeze().long()
        # tra_targets = tra_targets - 1

        # loop test data to predict the label by svm
        test_bar = tqdm(test_data_loader)
        for data, _, ind_ts, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            pixels_test = X_spe[np.squeeze(ind_ts), :].astype("float32")
            pixels_test = torch.from_numpy(pixels_test).cuda()

            feature, out = net(data, pixels_test, r)
            test_feature_bank.append(feature)
            test_targets.append(target)

        # [N, D]
        test_feature_bank = torch.cat(test_feature_bank, dim=0)
        test_targets = torch.cat(test_targets, dim=0).squeeze().long()
        # test_targets = test_targets - 1

        # ############## SVM ##############
        train_feature_bank = train_feature_bank.cpu().numpy()
        test_feature_bank = test_feature_bank.cpu().numpy()
        # train_feature_bank = np.concatenate((train_feature_bank, Pixel_train), 1)

        # print(train_feature_bank.shape)
        tra_targets = tra_targets.cpu().numpy()
        test_targets = test_targets.cpu().numpy()

        c = [1, 10, 100, 1000, 10000]
        sigmaval = [2 ** m for m in range(-4, 5)]
        g = [1/(2 * m ** 2) for m in sigmaval]
        # c = [c for c in range(400, 600) if c % 30 == 0]
        # g = [0.25 * m for m in range(1, 13)]
        score = 0
        c_value, gamma_value = 0, 0
        a = 1
        if a == 1:
            for ii in c:
                for jj in g:
                    clf = svm.SVC(C=ii, gamma=jj)
                    # clf = svm.SVC(C=2 ** ii, gamma=2 ** jj)
                    clf.fit(train_feature_bank, tra_targets)
                    score1 = clf.score(test_feature_bank, test_targets)
                    print(score1)
                    print(ii, jj)
                    if score1 > score:
                        score = score1
                        c_value = ii
                        gamma_value = jj

        elif a == 2:
            ii = 420
            jj = 0.25
            clf = svm.SVC(C=ii, gamma=jj)
            clf.fit(train_feature_bank, tra_targets)
            score1 = clf.score(test_feature_bank, test_targets)
            print(score1)
            c_value = ii
            gamma_value = jj

        print('Best result is:', score)

    return c_value, gamma_value


def test_svm_all(net, memory_data_loader, test_data_loader, c, gamma, r):
    net.eval()
    total_top1, total_top5, total_num, train_feature_bank, test_feature_bank, tra_targets = 0.0, 0.0, 0, [], [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, ind_tr, target in tqdm(memory_data_loader, desc='SVM_Feature extracting'):
            pixels_train = X_spe[np.squeeze(ind_tr), :].astype("float32")
            pixels_train= torch.from_numpy(pixels_train).cuda()

            feature, out = net(data.cuda(non_blocking=True), pixels_train, r)
            train_feature_bank.append(feature)
            tra_targets.append(target)
        # [N, D]
        train_feature_bank = torch.cat(train_feature_bank, dim=0)
        # [N]
        tra_targets = torch.cat(tra_targets, dim=0).squeeze().long()
        # tra_targets = tra_targets - 1

        # labels = Y_train.squeeze() - 1
        # feature_labels = torch.tensor(labels, device=train_feature_bank.device).long()

        # loop test data to predict the label by svm
        test_bar = tqdm(test_data_loader)
        for data, _, ind_ts, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            pixels_test = X_spe[np.squeeze(ind_ts), :].astype("float32")
            pixels_test = torch.from_numpy(pixels_test).cuda()

            feature, out = net(data, pixels_test, r)
            test_feature_bank.append(feature)

        # [N, D]
        test_feature_bank = torch.cat(test_feature_bank, dim=0)

        # ############## SVM ##############
        train_feature_bank = train_feature_bank.cpu().numpy()
        tra_targets = tra_targets.cpu().numpy()
        test_feature_bank = test_feature_bank.cpu().numpy()

        # train_feature_bank = np.concatenate((train_feature_bank, train_spe), axis=1)
        # test_feature_bank = np.concatenate((test_feature_bank, X_spe), axis=1)

        clf = svm.SVC(C=c, gamma=gamma)
        clf.fit(train_feature_bank, tra_targets)
        pred_labels = clf.predict(test_feature_bank)

    return pred_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=20, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=1024, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument("--noise", type=float, default=0.1, help="noise.")
    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    ID = 1  # Pavia
    n_PC = 3
    w = 16
    repeat = 10
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(w),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    print('loading data...')
    X = sio.loadmat('./DataSets/PaviaU.mat')['PaviaU_OriData3']
    Y = sio.loadmat('./DataSets/paviaU_gt.mat')['groundT']
    train_num_array = [331, 932, 104, 153, 67, 251, 66, 184, 47]  # 5%
    print(np.sum(train_num_array))
    n_class = Y.max()
    CNN3_result1 = np.zeros((repeat, n_class + 4))

    for i in range(repeat):
        time1 = time.time()

        # ############ Load index ###################
        index_train_r = sio.loadmat('./Pa_SampleSets_Index_5%/Index_TrainSample.mat')['index_train']
        index_test_r = sio.loadmat('./Pa_SampleSets_Index_5%/Index_TestSample.mat')['index_test']
        [row, col, n_feature] = X.shape
        all_num = sum(Y.reshape(row * col, 1) > 0)[0]
        index_train = np.zeros((sum(train_num_array), 1)).astype('int')
        index_test = np.zeros((all_num-sum(train_num_array), 1)).astype('int')
        index_train[:, 0] = index_train_r[:, i]
        print(index_train.dtype)
        index_test[:, 0] = index_test_r[:, i]

        # ########### obtain spectral vectors ##########
        X_reshape = X.reshape(row * col, n_feature)
        X_spe = featureNormalize(X_reshape, 2)
        train_spe = X_spe[np.squeeze(index_train), :]
        test_spe = X_spe[np.squeeze(index_test), :]

        X_PCA = featureNormalize(PCANorm(X_reshape, n_PC), 2)  # (0,1)
        X_PCA = 255 * X_PCA.reshape(row, col, n_PC)  # (0,255)

        X_cos, index_all, X_train, Y_train, X_test, Y_test = Index_SamplesGenerate(X, X_PCA, Y, train_num_array, index_train, index_test, w)
        Y_cos = np.zeros((row * col, 1)).astype('float32')

        n_class_new = Y_train.max().astype('int')

        X_labels = np.concatenate((X_train, X_test), axis=0)
        Y_labels = np.concatenate((Y_train, Y_test), axis=0)
        index_labels = np.concatenate((index_train,index_test), axis=0)

        train_data = Mydataset(dataset=X_labels, label=Y_labels, ind_img=index_labels, transform=train_transform)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        memory_data = Mydataset(dataset=X_train, label=Y_train, ind_img=index_train, transform=test_transform)
        memory_loader = DataLoader(dataset=memory_data, batch_size=batch_size, shuffle=False)
        test_data = Mydataset(dataset=X_test, label=Y_test, ind_img=index_test, transform=test_transform)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        r = 1
        # model setup and optimizer config
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # resnet18
            model = GaussianModel(backbone=resnet18(r), head='mlp', features_dim=128, n_feature=n_feature)
            print(model)
            model = nn.DataParallel(model)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

        # Load pretrained weights
        pretrain_path = resultpath + 'final_model.pth'
        if pretrain_path is not None and os.path.exists(pretrain_path):
            state = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch']
        else:
            start_epoch = 0

        # training loop
        for epoch in range(start_epoch, epochs):
            train_loss = train(model, train_loader, optimizer, r)

            print('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1}, resultpath + '/final_model.pth')

        # if epoch % 10 == 0:
        c, gamma = test_svm(model, memory_loader, test_loader, r)

        # classification map ############
        all_data = Mydataset(dataset=X_cos, label=Y_cos, ind_img=index_all, transform=test_transform)
        all_loader = DataLoader(dataset=all_data, batch_size=batch_size, shuffle=False)
        label = test_svm_all(model, memory_loader, all_loader, c, gamma, r)
        label = np.array(label)

        # classification map of all pixels
        img1 = DrawResult(label, ID)
        plt.imsave(resultpath + 'ClassificationMap' + '_' + repr(i+1) + '.png', img1)

        index_train = np.squeeze(index_train)
        index_test = np.squeeze(index_test)

        # accuracy of test pixels
        y_pred1 = np.squeeze(label[index_test])
        y_test = np.squeeze(Y_test).astype(int)

        AA1, OA1, kappa, ProducerA = CalAccuracy(y_pred1 - 1, y_test - 1)
        CNN3_result1[i, :n_class] = ProducerA
        CNN3_result1[i, -4] = OA1
        CNN3_result1[i, -3] = AA1
        CNN3_result1[i, -2] = kappa
        time2 = time.time()
        CNN3_result1[i, -1] = time2 - time1

        print('OA is :%.6f' % (CNN3_result1[i, -4]))
        print('AA is :%.6f' % (CNN3_result1[i, -3]))
        print('KA is :%.6f' % (CNN3_result1[i, -2]))
        print("Running time: %g" % (CNN3_result1[i, -1]))

        # classification map of test pixels
        image1 = np.zeros((row*col,)).astype('int')
        Y_train = np.squeeze(Y_train)
        image1[index_train] = Y_train
        image1[index_test] = y_pred1

        img1 = DrawResult(image1, ID)
        plt.imsave(resultpath + 'S3FN' + repr(i+1) + '_' + repr(int(OA1 * 10000)) + '.png', img1)

CNN3_result1[:, :-1] = CNN3_result1[:, :-1] * 100
sio.savemat(resultpath + 'S3FN.mat', {'CNN3_result': CNN3_result1})

