#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

import numpy as np
import torch.nn as nn
import torch
from model_service.pytorch_model_service import PTServingBaseService


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

        x = self.softmax(x)
        # x = torch.squeeze(x, 0)
        return x


class PredictService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PredictService, self).__init__(model_name, model_path)

        super_model = torch.load(model_path, map_location='cpu')

        model1 = Linear(384, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model2 = Linear(333, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model3 = Linear(95, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model4 = Linear(113, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model5 = Linear(103, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model6 = Linear(444, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model7 = Linear(246, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model8 = Linear(500, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)

        model9 = Linear(373, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model10 = Linear(335, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model11 = Linear(87, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model12 = Linear(109, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model13 = Linear(101, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model14 = Linear(395, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model15 = Linear(236, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)
        model16 = Linear(500, 1024, 512, 128, 64, 32, 3, dropout_p=0.5)

        model1.load_state_dict(super_model[0])
        model2.load_state_dict(super_model[1])
        model3.load_state_dict(super_model[2])
        model4.load_state_dict(super_model[3])
        model5.load_state_dict(super_model[4])
        model6.load_state_dict(super_model[5])
        model7.load_state_dict(super_model[6])
        model8.load_state_dict(super_model[7])

        model9.load_state_dict(super_model[8])
        model10.load_state_dict(super_model[9])
        model11.load_state_dict(super_model[10])
        model12.load_state_dict(super_model[11])
        model13.load_state_dict(super_model[12])
        model14.load_state_dict(super_model[13])
        model15.load_state_dict(super_model[14])
        model16.load_state_dict(super_model[15])

        model1.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        model5.eval()
        model6.eval()
        model7.eval()
        model8.eval()

        model9.eval()
        model10.eval()
        model11.eval()
        model12.eval()
        model13.eval()
        model14.eval()
        model15.eval()
        model16.eval()

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        self.model8 = model8

        self.model9 = model9
        self.model10 = model10
        self.model11 = model11
        self.model12 = model12
        self.model13 = model13
        self.model14 = model14
        self.model15 = model15
        self.model16 = model16

        self.load_preprocess()

    def load_preprocess(self, mean_name='mean.npy', std_name='std.npy'):
      dir_path = os.path.dirname(os.path.realpath(self.model_path))
      mean_path = os.path.join(dir_path, mean_name)
      std_path = os.path.join(dir_path, std_name)
      self.mean = np.load(mean_path)
      self.std = np.load(std_path)

    def _preprocess(self, data):
        print('pre_data:{}'.format(data))
        preprocessed_data = {}
        m = []
        for d in data:
            for k, v in data.items():
                for file_name, features_path in v.items():
                    x = np.load(features_path)
                    x = (x - self.mean) / self.std
                    x = np.nan_to_num(x)
                    x[x > 1000000] = 0
                    x[x < -1000000] = 0
                    # x = torch.from_numpy(x).to(torch.float32)
                    preprocessed_data[k] = x
                    m.append(x)
        return m

    def _inference(self, data):

        inputs = np.array(data)
        inputs = torch.from_numpy(inputs).to(torch.float32)

        inputs1 = inputs[:, 14327:14711]
        inputs2 = inputs[:, 14779:15112]
        inputs3 = inputs[:, 15195:15290]
        inputs4 = inputs[:, 15290:15403]
        inputs5 = inputs[:, 15403:15506]
        inputs6 = inputs[:, 16218:16662]
        inputs7 = inputs[:, 16733:16979]
        inputs8 = inputs[:, 21479:21979]

        inputs9 = inputs[:, 335:708]
        inputs10 = inputs[:, 755:1090]
        inputs11 = inputs[:, 1163:1250]
        inputs12 = inputs[:, 1250:1359]
        inputs13 = inputs[:, 1359:1460]
        inputs14 = inputs[:, 2096:2491]
        inputs15 = inputs[:, 2557:2793]
        inputs16 = inputs[:, 7290:7790]

        outputs1 = self.model1(inputs1)
        outputs2 = self.model2(inputs2)
        outputs3 = self.model3(inputs3)
        outputs4 = self.model4(inputs4)
        outputs5 = self.model5(inputs5)
        outputs6 = self.model6(inputs6)
        outputs7 = self.model7(inputs7)
        outputs8 = self.model8(inputs8)

        outputs9 = self.model9(inputs9)
        outputs10 = self.model10(inputs10)
        outputs11 = self.model11(inputs11)
        outputs12 = self.model12(inputs12)
        outputs13 = self.model13(inputs13)
        outputs14 = self.model14(inputs14)
        outputs15 = self.model15(inputs15)
        outputs16 = self.model16(inputs16)

        outputs1 = outputs1.cpu().detach().numpy().reshape(-1, 1)
        outputs2 = outputs2.cpu().detach().numpy().reshape(-1, 1)
        outputs3 = outputs3.cpu().detach().numpy().reshape(-1, 1)
        outputs4 = outputs4.cpu().detach().numpy().reshape(-1, 1)
        outputs5 = outputs5.cpu().detach().numpy().reshape(-1, 1)
        outputs6 = outputs6.cpu().detach().numpy().reshape(-1, 1)
        outputs7 = outputs7.cpu().detach().numpy().reshape(-1, 1)
        outputs8 = outputs8.cpu().detach().numpy().reshape(-1, 1)

        outputs9 = outputs9.cpu().detach().numpy().reshape(-1, 1)
        outputs10 = outputs10.cpu().detach().numpy().reshape(-1, 1)
        outputs11 = outputs11.cpu().detach().numpy().reshape(-1, 1)
        outputs12 = outputs12.cpu().detach().numpy().reshape(-1, 1)
        outputs13 = outputs13.cpu().detach().numpy().reshape(-1, 1)
        outputs14 = outputs14.cpu().detach().numpy().reshape(-1, 1)
        outputs15 = outputs15.cpu().detach().numpy().reshape(-1, 1)
        outputs16 = outputs16.cpu().detach().numpy().reshape(-1, 1)

        outputs1 = np.argmax(outputs1[:, 0])
        outputs2 = np.argmax(outputs2[:, 0])
        outputs3 = np.argmax(outputs3[:, 0])
        outputs4 = np.argmax(outputs4[:, 0])
        outputs5 = np.argmax(outputs5[:, 0])
        outputs6 = np.argmax(outputs6[:, 0])
        outputs7 = np.argmax(outputs7[:, 0])
        outputs8 = np.argmax(outputs8[:, 0])

        outputs9 = np.argmax(outputs9[:, 0])
        outputs10 = np.argmax(outputs10[:, 0])
        outputs11 = np.argmax(outputs11[:, 0])
        outputs12 = np.argmax(outputs12[:, 0])
        outputs13 = np.argmax(outputs13[:, 0])
        outputs14 = np.argmax(outputs14[:, 0])
        outputs15 = np.argmax(outputs15[:, 0])
        outputs16 = np.argmax(outputs16[:, 0])

        # 根据不同模型在验证集上F1 score表现的结果设置不同的权重占比并做majority voting(voting前删除了一些表现离群的图谱模型)
        res = [outputs1, outputs1, outputs1, outputs3, outputs3, outputs4, outputs4, outputs4, outputs4, outputs5,
               outputs5, outputs6, outputs6, outputs7, outputs7, outputs9, outputs9, outputs9, outputs9, outputs10,
               outputs10, outputs11, outputs11, outputs11, outputs11, outputs13, outputs13, outputs16, outputs16]

        mci_1 = np.argmax(np.bincount(res))
        if mci_1 == 0:
            pre_result = [1, 0, 0]
        elif mci_1 == 1:
            pre_result = [0, 1, 0]
        else:
            pre_result = [0, 0, 1]
        return pre_result

    def _postprocess(self, data):

        infer_output = {}
        infer_output['scores'] = data
        return infer_output

