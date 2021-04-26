from dataset import SlidingWindowDataset, MnistWindowDataset
from e3d_lstm import E3DLSTM
from functools import lru_cache
from torch.utils.data import DataLoader
from utils import h5_virtual_file, window, weights_init, MNISTdataLoader
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaxiBJTrainer(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float

        # TODO make all configurable
        self.num_epoch = 100
        self.batch_size = 16

        self.input_time_window = 10
        self.output_time_horizon = 5
        self.temporal_stride = 5
        self.temporal_frames = 5
        self.time_steps = (
            self.input_time_window - self.temporal_frames + 1
        ) // self.temporal_stride

        # Initiate the network
        # CxT×H×W
        input_shape = (1, self.temporal_frames, 64, 64)
        output_shape = (1, self.output_time_horizon, 64, 64)

        self.tau = 2
        hidden_size = 64
        kernel = (2, 5, 5)
        lstm_layers = 4

        self.encoder = E3DLSTM(
            input_shape, hidden_size, lstm_layers, kernel, self.tau
        ).type(dtype)
        self.decoder = nn.Conv3d(
            hidden_size, output_shape[0], kernel, padding=(0, 2, 2)
        ).type(dtype)
        # self.decoder = nn.Sequential(
        #   nn.Conv3d(hidden_size * self.time_steps, output_shape[0]),
        #  nn.ConvTranspose3d(output_shape[0], output_shape[0], kernel)
        # ).type(dtype)

        self.to(self.device)

        # Setup optimizer
        params = self.parameters(recurse=True)
        # TODO learning rate scheduler
        # Weight decay stands for L2 regularization
        self.optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0)

        self.apply(weights_init())

    def forward(self, input_seq):
        return self.decoder(self.encoder(input_seq))

    def loss(self, input_seq, target):
        output = self(input_seq)

        l2_loss = F.mse_loss(output * 255, target * 255)
        l1_loss = F.l1_loss(output * 255, target * 255)
        from utils import draw
        draw(target, output)

        return l1_loss, l2_loss

    @property
    @lru_cache(maxsize=1)
    def data(self):
        path = './mnist_test_seq.npy'
        raw_data = MNISTdataLoader(path)
        data = np.expand_dims(raw_data, axis=1)
        return data

    def get_trainloader(self, raw_data, shuffle=True):
        # NOTE note we do simple transformation, only approx within [0,1]
        dataset = MnistWindowDataset(
            raw_data,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def validate(self, val_dataloader):
        self.eval()

        sum_l1_loss = 0
        sum_l2_loss = 0
        with torch.no_grad():
            for i, (input, target) in enumerate(val_dataloader):
                frames_seq = []

                for indices in window(
                    range(self.input_time_window),
                    self.temporal_frames,
                    self.temporal_stride,
                ):
                    # batch x channels x time x window x height
                    frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])
                input = torch.stack(frames_seq, dim=0).to(self.device)
                target = target.to(self.device)

                l1_loss, l2_loss = self.loss(input, target)
                sum_l1_loss += l1_loss
                sum_l2_loss += l2_loss

        print(f"Validation L1:{sum_l1_loss / (i + 1)}; L2: {sum_l2_loss / (i + 1)}")

    def resume_train(self, ckpt_path=".master_ckpt/taxibj_trainer.pt", resume=False):
        # 2 weeks / 30min time step = 672
        train_dataloader = self.get_trainloader(self.data[:-672])
        val_dataloader = self.get_trainloader(self.data[-672:], False)

        if resume:
            checkpoint = torch.load(ckpt_path)
            epoch = checkpoint["epoch"]
            self.load_state_dict(checkpoint["state_dict"])
        else:
            epoch = 0

        while epoch < self.num_epoch:
            epoch += 1
            for i, (input, target) in enumerate(train_dataloader):
                frames_seq = []

                for indices in window(
                    range(self.input_time_window),
                    self.temporal_frames,
                    self.temporal_stride,
                ):
                    # batch x channels x time x window x height
                    frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])
                input = torch.stack(frames_seq, dim=0).to(self.device)
                target = target.to(self.device)
                self.train()
                self.optimizer.zero_grad()
                l1_loss, l2_loss = self.loss(input, target)
                loss = l1_loss + l2_loss
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    print(
                        "Epoch: {}/{}, step: {}/{}, mse: {}".format(
                            epoch, self.num_epoch, i, len(train_dataloader), l2_loss
                        )
                    )

            torch.save({"epoch": epoch, "state_dict": self.state_dict()}, ckpt_path)
            self.validate(val_dataloader)


if __name__ == "__main__":
    trainer = TaxiBJTrainer()
    trainer.resume_train()
