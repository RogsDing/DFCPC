import torch
import torch.nn as nn


class Feature_ex(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1_1 = nn.Sequential(nn.Conv2d(self.in_channels[0], 32, 3, 1, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=False))
        self.conv1_2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=False))
        self.conv2_1 = nn.Sequential(nn.Conv2d(self.in_channels[1], 32, 3, 1, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=False))
        self.conv2_2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=False))

    def forward(self, x):
        x_h = self.conv1_2(self.conv1_1(x[0]))
        x_l = self.conv2_2(self.conv2_1(x[1]))
        x_out = torch.cat((x_h, x_l), dim=1)
        return x_out     # x_h=256,64,7,7 x_l=256,64,7,7


class Attention(nn.Module):
    def __init__(self, inchannels, patch_size):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchannels, inchannels, 3, 1, 1),
                                   nn.BatchNorm2d(inchannels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(inchannels, inchannels, 5, 1, 2),
                                   nn.BatchNorm2d(inchannels),
                                   nn.ReLU())
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mlp1 = nn.Conv2d(128, 128, 1)
        self.mlp2 = nn.Conv2d(1, 1, 1)
        self.mlp3 = nn.Sequential(nn.Linear(128+patch_size*patch_size, 64),
                                  nn.GELU(),
                                  nn.Dropout(0.1),
                                  nn.Linear(64, 128))
        self.mlp4 = nn.Sequential(nn.Linear(128+patch_size*patch_size, patch_size*patch_size),
                                  nn.GELU(),
                                  nn.Dropout(0.1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.size()
        x_1 = self.conv1(x)      # 构造不同尺度的特征输入
        x_2 = self.conv2(x)

        x1_1 = self.mp(x_1)      # 256,128,1,1
        x1_2 = self.ap(x_1)      # 256,128,1,1
        x1_1 = self.mlp1(x1_1)
        x1_2 = self.mlp1(x1_2)
        x1 = x1_1 + x1_2
        x1 = x1.reshape(x1.size(0), x1.size(1), -1)     # B,C,1
        x1 = x1.permute(0, 2, 1)    # B,1,C

        x2_1 = torch.mean(x_2, dim=1, keepdim=True)  # 256,1,7,7
        x2_2, _ = torch.max(x_2, dim=1, keepdim=True)  # 256,1,7,7
        x2_1 = self.mlp2(x2_1)
        x2_2 = self.mlp2(x2_2)
        x2 = x2_1 + x2_2
        x2 = x2.reshape(x2.size(0), x2.size(1), -1)     # B,1,h*w

        x_add = torch.cat((x1, x2), dim=-1)     # B,1,(h*w+C)
        x1 = self.mlp3(x_add)       # B,1,C
        x2 = self.mlp4(x_add)       # B,1,h*w

        x1 = x1.permute(0, 2, 1)    # B,C,1
        x1 = x1.reshape(x1.size(0), x1.size(1), 1, 1)     # B,C,1,1
        x1 = self.sigmoid(x1)
        x1 = x1 * x_1

        x2 = x2.reshape(x2.size(0), x2.size(1), h, w)
        x2 = self.sigmoid(x2)
        x2 = x2 * x1_2
        return x1, x2


class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ContrastiveHead, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp_head(x)
        return x


class ClusteringHead(nn.Module):
    def __init__(self, n_dim, n_class, alpha=1.):
        super(ClusteringHead, self).__init__()
        # Clustering head
        self.alpha = alpha
        # initial_cluster_centers = torch.tensor(torch.randn((n_class, n_dim), dtype=torch.float, requires_grad=True))
        self.cluster_centers = nn.Parameter(torch.Tensor(n_class, n_dim), requires_grad=True)
        # torch.nn.init.orthogonal_(self.cluster_centers.data, gain=1)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        """
        :param x: n_batch * n-dim
        :return:
        """
        pred_prob = self.get_cluster_prob(x)
        return pred_prob

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class Net(nn.Module):
    def __init__(self,n_modalities, in_channels, in_patch_size, common_channel, n_class, dim_emebeding):   # 2,{63,2},{7,7},32,6,512
        super(Net, self).__init__()
        self.conv = Feature_ex(in_channels)
        self.attention = Attention(128, in_patch_size[0])
        self.clustering_head = ClusteringHead(dim_emebeding, n_class, alpha=1) ## ContrastiveHead(512, 128)
        self.b1 = nn.Sequential(nn.Conv2d(128, dim_emebeding, 1),
                                nn.BatchNorm2d(dim_emebeding),
                                nn.ReLU(inplace=False))
        self.b2 = nn.AdaptiveAvgPool2d(1)  # nn.AdaptiveAvgPool2d(output_size) 中的 output_size 参数表示期望的输出大小
        self.clustering_head = ClusteringHead(dim_emebeding, n_class, alpha=1)  ## ContrastiveHead(512, 128)

    def forward(self, x_1, x_2):
        """
        :param x_1, x_2: tuple of modalities, e.g., [aug_1, aug_2]-->
        ([img_rgb, img_hsi, img_sar], [img_rgb, img_hsi, img_sar])
        :return:
        """
        x_11, x_12 = self.attention(self.conv(x_1))  # # concatenated modalities: [batch, n_channel, width, 2*height]
        x_21, x_22 = self.attention(self.conv(x_2))
        x_1 = x_11 + x_12
        x_1 = self.b2(self.b1(x_1))
        x_1 = x_1.view(x_1.size(0), -1)  # (256,256)
        x_2 = x_21 + x_22
        x_2 = self.b2(self.b1(x_2))
        x_2 = x_2.view(x_2.size(0), -1)

        y_1 = self.clustering_head(x_1)
        y_2 = self.clustering_head(x_2)

        return y_1, y_2

    def forward_embedding(self, x):
        # h = self.clustering_head(self.vit(self.embedding_layer(x)))
        h1, h2 = self.attention(self.conv(x))
        h = h1 + h2
        h = self.b2(self.b1(h))
        h = h.view(h.size(0), -1)
        return h

    def forward_cluster(self, x, return_h=False):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        h1, h2 = self.attention(self.conv(x))
        h = h1 + h2
        h = self.b2(self.b1(h))
        h = h.view(h.size(0), -1)
        pred = self.clustering_head(h)
        labels = torch.argmax(pred, dim=1)
        if return_h:
            return labels, h
        return labels