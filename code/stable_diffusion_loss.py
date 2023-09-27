"""
===============================================================================
========================You can run this script directly=======================
===============================================================================

This is the model definition of the sky generation model
For more detail about the design, please refer to the report

You can get the output like this:
    batch size: 5
    predict image shape: torch.Size([5, 4, 32, 32])
    real image shape:torch.Size([5, 4, 32, 32])
    mask shape:torch.Size([5, 32, 32])
    loss: 1.9836117029190063

===============================================================================
===============================================================================
"""

import torch
import torch.nn.functional as F


# input a batch
# m and n are the super parameters, the default value is 1
def our_loss(predict_batch, real_batch, mask_batch, m=1, n=1):
    # get the batch size
    batch_size = predict_batch.shape[0]

    # the total loss of this batch
    total_loss = 0

    for i in range(batch_size):

        # expand the shape of the mask to 4,32,32
        now_mask = mask_batch[i].unsqueeze(0).repeat(4, 1, 1)

        # compute the mask of different regions
        sky_dim = now_mask.sum()
        total_dim = now_mask.numel()
        ground_dim = total_dim - sky_dim

        # compute the percent of masks
        sky_percent = sky_dim / total_dim
        ground_percent = ground_dim / total_dim

        # sometimes the dimension might be 0
        if sky_dim == 0:
            sky_loss = 0
        else:
            sky_loss = F.mse_loss(predict_batch[i] * now_mask, real_batch[i] * now_mask, reduction='mean')
            sky_loss = sky_loss.sum() / sky_dim

        if ground_dim == 0:
            ground_loss = 0
        else:
            ground_loss = F.mse_loss(predict_batch[i] * (1 - now_mask), real_batch[i] * (1 - now_mask), reduction='none')
            ground_loss = ground_loss.sum() / ground_dim

        # compute the weighted loss
        loss = ground_loss * (1 + ground_percent * m) + sky_loss * (1 + sky_percent * n)
        total_loss += loss

    # compute the average loss of each sample
    return total_loss / batch_size


if __name__ == '__main__':
    batch_size = 5
    predict = torch.randn(batch_size, 4, 32, 32)
    real = torch.randn(batch_size, 4, 32, 32)
    mask = torch.ones(batch_size, 32, 32)

    # add some noise
    mask[0][0][0] = 0
    mask[1][0][0] = 0
    mask[2][0][0] = 0
    mask[3][0][0] = 0
    mask[4][0][0] = 0

    loss = our_loss(predict, real, mask, 1, 1)
    print(f'batch size: {batch_size}')
    print(f"predict image shape: {predict.shape}")
    print(f"real image shape:{real.shape}")
    print(f"mask shape:{mask.shape}")
    print(f"loss: {loss}")
