# coding: utf-8
import sys
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch

from common.plot import print_images

sys.path.append(os.pardir)  # 부모 디렉터리 파일을 가져올 수 있도록 설정


class CNN:
    """단순한 합성곱 신경망

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_size : 입력 크기（3 * 48 * 48）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 43
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 0.01
    """

    def __init__(
        self,
        input_dim=(3, 48, 48),
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=43,
        weight_init_std=0.1,
        visualize=False,
        device="cpu",
    ):
        # Confusion Matrix 연산을 위한 label 수 저장.
        self.output_size = output_size
        self.confusion_matrix = None
        self.device = device
        self.visualize = visualize

    def predict(self, x, train_flg=True, is_final=False):
        x = x.to(self.device)
        layer_to_visualize = ["conv", "pool"]
        x_to_visualize = OrderedDict() if is_final else None

        for layer_name, layer_value in self.layers.items():
            if "bNorm" in layer_name:
                x = layer_value.forward(x, train_flg=train_flg)
            else:
                x = layer_value.forward(x)

            if (
                self.visualize
                and is_final
                and any(x.lower() in layer_name.lower() for x in layer_to_visualize)
            ):
                x_to_visualize[layer_name] = x[0][:10].cpu().numpy()

        if self.visualize and is_final:
            print_images(x_to_visualize)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # accuracy, f1score를 return 하는 함수.
    def accuracy_f1score(self, x, t, batch_size=100, is_final=False):
        # x : data
        # t : label

        x = x.to(self.device)
        t = t.to(self.device)

        # one hot label -> normal label
        if t.ndim != 1:
            t = torch.argmax(t, dim=1)

        labels = self.output_size

        confusion_matrix = np.zeros((labels, labels))

        # range(train data 개수 / batch_size)
        for i in range(int(x.shape[0] / batch_size)):
            # i번째 batch의 data list
            tx = x[i * batch_size : (i + 1) * batch_size]
            # i번째 batch의 label list
            tt = t[i * batch_size : (i + 1) * batch_size].cpu().numpy()

            # 매 batch당 classification
            y = self.predict(tx, is_final=(is_final and i == 0)).cpu().numpy()
            y = np.argmax(y, axis=1)

            # confusion matrix
            for j in range(len(y)):
                confusion_matrix[tt[j]][y[j]] += 1

        # accuracy(맞은 것 세기)
        accuracy = 0.0
        for i in range(labels):
            accuracy += confusion_matrix[i][i]
        accuracy = accuracy / x.shape[0]

        # precision
        precision = [0] * labels
        precision_devider = np.sum(confusion_matrix, axis=0)

        # TP + FP
        for i in range(labels):
            # 0으로 나누는 것 방지. 우선은 그냥 0으로 뒀음.
            if precision_devider[i] == 0:
                precision[i] = 0
            else:
                precision[i] = confusion_matrix[i][i] / precision_devider[i]
        precision_avg = np.mean(precision)

        # recall
        recall = [0] * labels
        recall_devider = np.sum(confusion_matrix, axis=1)

        # TP + FN
        for i in range(0, labels):
            # 0으로 나누는 것 방지.
            if recall_devider[i] == 0:
                recall[i] = 0
            else:
                recall[i] = confusion_matrix[i][i] / recall_devider[i]
        recall_avg = np.mean(recall)

        # f1 score
        f1score = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)

        return accuracy, f1score, confusion_matrix

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params["W" + str(i + 1)]
            self.layers[key].b = self.params["b" + str(i + 1)]
