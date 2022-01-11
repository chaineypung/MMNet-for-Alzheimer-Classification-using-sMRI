import argparse
import os
import sys
import numpy as np
import warnings
import pandas as pd
from pandas.errors import EmptyDataError
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import random
from sklearn.metrics import recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def load_data(subject_dir, csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    subjects = os.listdir(subject_dir)

    x = []
    y = []
    for subject in subjects:
        features_path = os.path.join(subject_dir, subject)
        if not os.path.exists(features_path) or not features_path.endswith('npy'):
            continue
        else:
            row = df.loc[subject.split('.')[0]]
            label = int(row['Label'])

            x.append(np.load(features_path))
            y.append(label)

    x = np.array(x)
    y = np.array(y)
    return x, y


class MyDataset(data.Dataset):
    def __init__(self, x, y, device):
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y)
        self.device = device

    def __getitem__(self, index):
        xi = self.x[index].to(self.device)
        yi = self.y[index].to(self.device)
        return xi, yi

    def __len__(self):
        return len(self.y)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Linear(nn.Module):
    def __init__(self, in_dim,
                 n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5,
                 out_dim, dropout_p=0.):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1, bias=True)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2, bias=False)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3, bias=False)
        self.layer4 = nn.Linear(n_hidden_3, n_hidden_4, bias=False)
        self.layer5 = nn.Linear(n_hidden_4, n_hidden_5, bias=False)
        self.layer6 = nn.Linear(n_hidden_5, out_dim)
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        # self.relu = nn.Sigmoid()
        # self.relu = nn.Threshold(0.01, 0)
        self.dropout0 = nn.Dropout(p=0.3)
        self.dropout = nn.Dropout(p=dropout_p)
        self.softmax = nn.Softmax(dim=1)
        self.batchnorm1 = nn.BatchNorm1d(1)
        self.batchnorm2 = nn.BatchNorm1d(1)
        self.batchnorm3 = nn.BatchNorm1d(1)
        self.batchnorm4 = nn.BatchNorm1d(1)
        self.batchnorm5 = nn.BatchNorm1d(1)

    def forward(self, x):
        # x = torch.unsqueeze(x, 0)

        x = self.layer1(x)
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.batchnorm1(x)
        x = torch.squeeze(x, 1)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.batchnorm2(x)
        x = torch.squeeze(x, 1)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.batchnorm3(x)
        x = torch.squeeze(x, 1)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.batchnorm4(x)
        x = torch.squeeze(x, 1)
        x = self.dropout(x)
        x = self.layer5(x)
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.batchnorm5(x)
        x = torch.squeeze(x, 1)
        x = self.dropout(x)
        x = self.layer6(x)

        # x = self.softmax(x)
        # x = torch.squeeze(x, 0)
        return x


# 创建解析
parser = argparse.ArgumentParser(description="AD Detector", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 添加参数
parser.add_argument('--train_url', type=str, default='obs://icloud-bucket/baseline/model',
                    help='the path model saved')
parser.add_argument('--data_url', type=str, default='obs://icloud-bucket/baseline/train_data', help='the training data')
parser.add_argument('--epochs', type=int, default=2500, help='max epoch')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--early_stop', type=int, default=100, help='early stop patience')


# parser.add_argument('--train_url', type=str, default='./fake_file',
#                     help='the path model saved')
# parser.add_argument('--data_url', type=str, default='./train_data', help='the training data')
# parser.add_argument('--epochs', type=int, default=2500, help='max epoch')
# parser.add_argument('--batch_size', type=int, default=256, help='batch size')
# parser.add_argument('--learning_rate', type=int, default=0.01, help='learning rate')
# parser.add_argument('--early_stop', type=int, default=100, help='early stop patience')

# 解析参数
args, unkown = parser.parse_known_args()
n_epochs = args.epochs
data_path = args.data_url
model_path = args.train_url
batch_size = args.batch_size
LR = args.learning_rate
early_stop = args.early_stop

# 预处理
traindata_path = os.path.join(data_path, "train")
labeldata_path = os.path.join(data_path, "train_open.csv")
train_x, label = load_data(traindata_path, labeldata_path)
train_x = np.nan_to_num(train_x, nan=0.0, posinf=0, neginf=0)
mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x - mean) / std
train_x = np.nan_to_num(train_x, nan=0.0, posinf=0, neginf=0)

# 挑出不同类别的图谱
atlas1 = list(range(14327, 14711))
atlas2 = list(range(14779, 15112))
atlas3 = list(range(15195, 15290))
atlas4 = list(range(15290, 15403))
atlas5 = list(range(15403, 15506))
atlas6 = list(range(16218, 16662))
atlas7 = list(range(16733, 16979))
atlas8 = list(range(21479, 21979))
atlas9 = list(range(335, 708))
atlas10 = list(range(755, 1090))
atlas11 = list(range(1163, 1250))
atlas12 = list(range(1250, 1359))
atlas13 = list(range(1359, 1460))
atlas14 = list(range(2096, 2491))
atlas15 = list(range(2557, 2793))
atlas16 = list(range(7290, 7790))
list_atlas = [atlas1, atlas2, atlas3, atlas4, atlas5, atlas6, atlas7, atlas8, atlas9, atlas10, atlas11, atlas12,
              atlas13, atlas14, atlas15, atlas16]

# 创建模型列表
MODEL = []

for atlas in range(len(list_atlas)):
    train_x_atlas = train_x[:, list_atlas[atlas]]
    train_y_atlas = label

    train_nc_x = train_x_atlas[label == 0, :]
    train_nc_y = train_y_atlas[label == 0]

    train_mci_x = train_x_atlas[label == 1, :]
    train_mci_y = train_y_atlas[label == 1]

    train_ad_x = train_x_atlas[label == 2, :]
    train_ad_y = train_y_atlas[label == 2]

    # 重采样,使三组不同类别样本数均衡
    test_x = np.concatenate((train_nc_x[748:, :], train_mci_x[1115:, :], train_ad_x[637:, :]), axis=0)
    test_y = np.concatenate((train_nc_y[748:], train_mci_y[1115:], train_ad_y[637:]), axis=0)

    train_nc_x = np.concatenate((train_nc_x[:748, :], train_nc_x[448:748, :]), axis=0)
    train_nc_y = np.concatenate((train_nc_y[:748], train_nc_y[448:748]), axis=0)

    train_ad_x = np.concatenate((train_ad_x[:637, :], train_ad_x[237:637, :]), axis=0)
    train_ad_y = np.concatenate((train_ad_y[:637], train_ad_y[237:637]), axis=0)

    train_x_atlas = np.concatenate((train_nc_x, train_mci_x, train_ad_x), axis=0)
    train_y_atlas = np.concatenate((train_nc_y, train_mci_y, train_ad_y), axis=0)

    index = [i for i in range(len(train_x_atlas))]
    random.shuffle(index)
    train_x_atlas = train_x_atlas[index, :]
    train_y_atlas = train_y_atlas[index]

    dataset = MyDataset(train_x_atlas, train_y_atlas, torch.device("cpu:0"))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_test = MyDataset(test_x, test_y, torch.device("cpu:0"))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model = Linear(len(list_atlas[atlas]), 1024, 512, 128, 64, 32, 3, dropout_p=0.5).to(device)

    adjust = 0
    optimizer = optim.SGD(model.parameters(), lr=LR * pow(0.5, adjust), momentum=0.7)
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    F1 = []
    early_stopping = EarlyStopping(patience=early_stop, verbose=True)

    for epoch in range(1, n_epochs + 1):
        if epoch % 100 == 0:
            adjust = adjust + 1
        for i, data in enumerate(data_loader, 0):
            model.train()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(device)
            # 训练模型
            optimizer.zero_grad()
            outputs0 = model(inputs)
            loss = criterion(outputs0, labels)
            loss.backward()
            optimizer.step()

            # 输出模型当前状态
            train_losses.append(loss.item())

            model.eval()
            for j, data in enumerate(data_loader_test, 0):
                inputs, labels1 = data
                inputs = inputs.to(device)
                labels1 = torch.tensor(labels1, dtype=torch.long)
                labels1 = labels1.to(device)
                outputs1 = model(inputs)
                loss1 = criterion(outputs1, labels1)
                loss_test = loss1
                valid_losses.append(loss_test.item())

                result = torch.max(outputs1, 1)[1].view(labels1.size())
                corrects = (result.data == labels1.data).sum().item()
                accuracy = corrects * 100.0 / len(labels1)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            epoch_len = len(str(n_epochs))
            atlas_len = len(str(len(list_atlas)))
            print_msg = (f'Atlas: [{atlas:>{atlas_len}}/{len(list_atlas):>{atlas_len}}] ' +
                         f'progress: [{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f} ' +
                         f'valid_acc: {accuracy:.1f}% ' +
                         f"valid_F1score:{f1_score(result.cpu().detach().numpy(), labels1.cpu().detach().numpy(), labels=[0, 1, 2], average='macro'):.5f}")
            print(print_msg)

            train_losses = []
            valid_losses = []
            F1.append(f1_score(result.cpu().detach().numpy(), labels1.cpu().detach().numpy(), labels=[0, 1, 2],
                               average='macro'))

            # early_stopping(valid_loss, model)
            early_stopping(1 - f1_score(result.cpu().detach().numpy(), labels1.cpu().detach().numpy(), labels=[0, 1, 2],
                                        average='macro'), model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    submodel = model.state_dict()
    MODEL.append(submodel)

model = {0: MODEL[0], 1: MODEL[1], 2: MODEL[2], 3: MODEL[3], 4: MODEL[4], 5: MODEL[5], 6: MODEL[6], 7: MODEL[7],
         8: MODEL[8], 9: MODEL[9], 10: MODEL[10], 11: MODEL[11], 12: MODEL[12], 13: MODEL[13], 14: MODEL[14],
         15: MODEL[15]}
m_path = os.path.join(model_path, "model.pth")
torch.save(model, m_path)
print("Model saved!")
