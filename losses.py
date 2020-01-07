import torch.nn as nn


def weighted_l1(poses, outputs):
    loss_weights = [0.5, 0.5]
    loss = nn.SmoothL1Loss()

    loss_d = loss(poses[:, 0], outputs[:, 0])
    loss_theta = loss(poses[:, 1], outputs[:, 1])
    loss_total = loss_d * loss_weights[0] + loss_theta * loss_weights[1]

    return loss_total