# coding: utf-8
import numpy as np
import torch

from common.functions import sigmoid, softmax, cross_entropy_error
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.clone().detach()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x.type("torch.DoubleTensor").to(self.W.get_device())

        out = torch.matmul(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = torch.matmul(dout, self.W.T)
        self.dW = torch.matmul(self.x.T, dout)
        self.db = torch.sum(dout, dim=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=0):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.clone().detach()
            dx[torch.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.rng = np.random.default_rng(43)

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = self.rng.random(size=x.shape) > self.dropout_ratio
            self.mask = torch.from_numpy(self.mask).to(x.get_device())

            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # 합성곱 계층은 4차원, 완전연결 계층은 2차원

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = torch.zeros(D, device=x.get_device())
            self.running_var = torch.zeros(D, device=x.get_device())
            # 위의 두 줄 기존 np.zeros(D) 텐서 대응

        if train_flg or self.batch_size is None:
            mu = torch.mean(x, dim=0)  # x.mean(axis=0)
            xc = x - mu
            var = torch.mean(xc**2, dim=0)  # np.mean(xc**2, axis=0)
            std = torch.sqrt(var + 10e-7)  # np.sqrt(var + 10e-7)
            xn = xc / std
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            if train_flg:
                self.running_mean = (
                    self.momentum * self.running_mean + (1 - self.momentum) * mu
                )
                self.running_var = (
                    self.momentum * self.running_var + (1 - self.momentum) * var
                )
        else:
            xc = x - self.running_mean
            xn = xc / (torch.sqrt(self.running_var + 10e-7))  # np.sqrt

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = torch.sum(self.xn * dout, dim=0)  # np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -torch.sum((dxn * self.xc) / (self.std * self.std), dim=0)
        # -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = torch.sum(dxc, dim=0)  # np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 중간 데이터（backward 시 사용）
        self.x = None
        self.col = None
        self.col_W = None

        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T.type("torch.FloatTensor").to(col.get_device())

        out = torch.matmul(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).permute(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.permute(0, 2, 3, 1).reshape(-1, FN)

        self.db = torch.sum(dout, dim=0)
        self.dW = torch.matmul(self.col.T.double(), dout.double())
        # 일단 버그나서 고쳐봄
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = torch.matmul(dout.double(), self.col_W.T.double())
        # 마찬가지로 일단 버그나서 타입 맞춰줬음.
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = torch.argmax(col, dim=1)
        out = torch.max(col, dim=1)[0]
        out = out.reshape(N, out_h, out_w, C).permute(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.permute(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w

        dmax = torch.zeros(
            (dout.cpu().numpy().size, pool_size), device=dout.get_device()
        )
        dmax[
            torch.arange(self.arg_max.cpu().numpy().size), self.arg_max.flatten()
        ] = dout.flatten().float()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
