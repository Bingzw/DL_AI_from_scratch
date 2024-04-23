import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0, "Temperature should be greater than 0"

        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)
        # The MLP for g(.) in the SimCLR paper
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        """
        images shape is (batch_size*n_views, C, H, W), concat the second augumentation
        of all batches after the first augmentation of all batches, thus the pos_mask need to be shifted by
        batch_size//2.
        For example, if batch_size = 3, then
        imgs = ((a_augmented_1, b_augmented_1, c_augmented_1), (a_augmented_2, b_augmented_2, c_augmented_2))))
        where a, b, c are the original images, then after concatenation,
        imgs = (a_augmented_1, b_augmented_1, c_augmented_1, a_augmented_2, b_augmented_2, c_augmented_2)
        then the pos_mask is
        tensor([[False, False, False, True, False, False],
                [False, False, False, False, True, False],
                [False, False, False, False, False, True],
                [True, False, False, False, False, False],
                [False, True, False, False, False, False],
                [False, False, True, False, False, False]])
        """
        imgs,_ = batch  # imgs shape is ((batch_size, c, H, W), (batch_size, c, H, W)) for n_views=2
        imgs = torch.cat(imgs, dim=0)
        # encode all images
        feats = self.convnet(imgs)  #
        # calculate the cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)  # return a n*n matrix with (i, j)
        # being the cosine similarity between i-th and j-th features
        # Mask out the diagonal (i.e. the similarity between the same feature)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -1e15)  # Set the diagonal to a very low value
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)  # divide 2 here because the n_views=2 (important)
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=1)
        nll = nll.mean()
        # Logging loss
        self.log(mode + '_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:, None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + '_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode + '_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode + '_acc_mean_pos', 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='val')





