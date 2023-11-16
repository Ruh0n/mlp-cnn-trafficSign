import itertools
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import numpy as np

from common.etc import print_gpu_info
from common.trainer import Trainer

from data.load_data import load_data
from models.CNN import CNN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_info(device)

    # Hyper Parameters
    BATCH_SIZES = [16]
    LRS = [0.001]

    for (batch, lr) in itertools.product(BATCH_SIZES, LRS):

        N = 6400
        EPOCHS = 30
        BATCH_SIZE = batch
        LR = lr
        # <data shape>
        # train     (N,     3,48,48)
        # validation(N,     3,48,48)
        # test      (7766,  3,48,48)
        # N=none 일 경우 각 90601, 31063, 7766개

        x_train, y_train, x_validation, y_validation, x_test, y_test = \
            load_data(N, _print=True, device=device)

        # Number of nodes in the previous layer
        # nodes_num = BATCH_SIZE*3*48*48
        # he = np.sqrt(2.0 / nodes_num)

        network = CNN(
            input_dim=(3, 48, 48),
            conv_param={"filter_num": 100, "filter_size": 5, "pad": 0, "stride": 1},
            hidden_size=100,
            output_size=43,
            weight_init_std=0.01,
            # 0.01, He
            device=device,
        )

        trainer = Trainer(
            network=network,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,

            epochs=EPOCHS,
            mini_batch_size=BATCH_SIZE,
            optimizer="sgd",
            optimizer_param={"lr": LR},
            evaluate_sample_num_per_epoch=1000,
            device=device,
        )
        trainer.train()

        # Accuracy Graph
        markers = {"train": "o", "test": "s"}
        plt.title(f"N: {N}, BATCH: {BATCH_SIZE}, LR: {LR}")
        x = torch.arange(EPOCHS)
        plt.plot(x, trainer.train_acc_list, marker="o", label="train", markevery=2)
        plt.plot(x, trainer.test_acc_list, marker="s", label="test", markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc="lower right")

        plt.show()

        # Confusion Matrix
        plt.figure(figsize=(20, 20))
        plt.matshow(trainer.confusion_matrix, fignum=1, cmap=plt.colormaps["Blues"])
        for (x, y), value in np.ndenumerate(trainer.confusion_matrix):
            plt.text(x, y, f"{int(value) if value != 0 else ' '}", va="center", ha="center")
        plt.show()
