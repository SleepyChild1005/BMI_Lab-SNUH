import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 512
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(672, 36, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        print('H1', H.shape)
        H = self.feature_extractor_part2(H)
        print('H2', H.shape)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part3(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        # Y_prob = self.classifier(Z)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Z

    # AUXILIARY METHODS


class Attention2(nn.Module):
    def __init__(self):
        super(Attention2, self).__init__()
        self.M = 512
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.squeeze(0)
        print('x',x.shape)

        H = self.feature_extractor_part1(x)
        print('H1', H.shape)
        H = self.feature_extractor_part2(H)
        print('H2', H.shape)
        H = H.view(-1, 50 * 4 * 4)
        print('H_m', H.shape)
        H = self.feature_extractor_part3(H)  # KxM
        print('H3', H.shape)

        A = self.attention(H)  # KxATTENTION_BRANCHES
        print('A', A.shape)
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        # Y_prob = self.classifier(Z)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Z


def run(device):
    input_tensor = torch.rand([250,1,28,28])
    model = Attention2()
    out = model.forward(input_tensor)
    print('out', out.shape)


    return input_tensor

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run(device)