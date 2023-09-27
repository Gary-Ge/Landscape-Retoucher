import torch
import torch.nn.functional as F

import pdb


# 输入的是batch
# m 和 n为可调节的超参数，默认为1
def our_loss(device, predict_batch, real_batch, mask_batch, m=0, n=0):
    # 输入的东西转tensor
    m = torch.tensor(m, dtype=torch.float)
    n = torch.tensor(n, dtype=torch.float)

    predict_batch.to(device)
    real_batch.to(device)
    mask_batch.to(device)
    m.to(device)
    n.to(device)

    # 先获取到batch size
    batch_size = predict_batch.shape[0]

    # 统计总loss
    total_loss = torch.tensor(0, dtype=torch.float).to(device)

    # 遍历batch里的每个样本
    for i in range(batch_size):
        # 扩充mask的通道到 4 32 32
        now_mask = mask_batch[i].unsqueeze(0).repeat(4, 1, 1)

        # 计算mask不同区域面积
        sky_dim = now_mask.sum()
        total_dim = now_mask.numel()
        ground_dim = total_dim - sky_dim

        # 计算mask比例
        sky_percent = sky_dim / total_dim
        ground_percent = ground_dim / total_dim

        # 特殊处理，可能dim为0
        if sky_dim == 0:
            sky_loss = torch.tensor(0, dtype=torch.float)
        else:
            sky_loss = F.mse_loss(predict_batch[i] * now_mask, real_batch[i] * now_mask, reduction='mean')
            #sky_loss = sky_loss.sum() / sky_dim

        if ground_dim == 0:
            ground_loss = torch.tensor(0, dtype=torch.float)
        else:
            ground_loss = F.mse_loss(predict_batch[i] * (torch.tensor(1) - now_mask), real_batch[i] * (torch.tensor(1) - now_mask), reduction='mean')
            #ground_loss = ground_loss.sum() / ground_dim
        

        # 计算最终loss
        #loss = ground_loss * ground_percent * (torch.tensor(1) + ground_percent * m) + sky_loss * sky_percent * (torch.tensor(1) + sky_percent * n)
        loss = ground_loss  * (torch.tensor(1) + ground_percent * m) + sky_loss * (torch.tensor(1) + sky_percent * n)
        total_loss += loss

    # 跟原版loss一样，平均batch里每个样本的loss
    return total_loss / batch_size


if __name__ == '__main__':
    batch_size = 5
    predict = torch.randn(batch_size, 4, 32, 32)
    real = torch.randn(batch_size, 4, 32, 32)
    mask = torch.ones(batch_size, 32, 32)

    # mask不要全部为1或0
    mask[0][0][0] = 0
    # mask[1][0][0] = 0
    # mask[2][0][0] = 0
    # mask[3][0][0] = 0
    # mask[4][0][0] = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss = our_loss(device, predict, real, mask, 1, 1)
    print(loss)
