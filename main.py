import torch

from common.etc import print_gpu_info
from common.plot import confusion_matrix, accuracy_graph, loss_graph
from common.trainer import Trainer
from data.load_data import load_data
from models.SimpleConvNet import SimpleConvNet
from models.VGG16 import VGG16


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_info(device)

    # Hyper Parameters - Fix
    N = 640
    BATCH_SIZE = 64

    # Hyper Parameters - Flex
    LR = 0.0001
    EPOCHS = 20
    CONV_FILTER_NUM = 100
    PADDING = 0
    HIDDEN_SIZE = 100

    # Load data
    datas = load_data(N_=N, print_=False)

    network = VGG16(
        conv_param={
            "filter_num": CONV_FILTER_NUM,
            "filter_size": 3,
            "pad": PADDING,
            "stride": 1,
        },
        hidden_size=HIDDEN_SIZE,
        weight_init_std=0.01,
        visualize=False,
        device=device,
    )

    trainer = Trainer(
        network=network,
        datas=datas,
        epochs=EPOCHS,
        mini_batch_size=BATCH_SIZE,
        optimizer="Adam",
        optimizer_param={"lr": LR},
        evaluate_sample_num_per_epoch=1000,
        device=device,
    )
    trainer.train()

    # Accuracy Graph
    accuracy_graph(trainer, EPOCHS)

    confusion_matrix(trainer)

    loss_graph(trainer, EPOCHS)
