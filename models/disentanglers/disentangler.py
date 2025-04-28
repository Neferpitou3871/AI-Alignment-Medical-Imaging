import torch
from torch.optim import Adam
import pytorch_lightning as pl

class Disentangler(pl.LightningModule):
    def __init__(self, mine_net, perturb_net, embedder, lambda_hyper=1e-3, lr=1e-4, weight_decay=1e-5):
        super(Disentangler, self).__init__()
        self.mine_net = mine_net
        self.perturb_net = perturb_net
        self.embedder = embedder
        self.embedder.eval()
        for param in self.embedder.parameters():
            param.requires_grad = False
            
        self.lambda_hyper = lambda_hyper
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, images, a):
        with torch.no_grad():
            z0 = self.embedder(images)
        phi_z0 = self.perturb_net(z0)
        return self.mine_net(phi_z0, a)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, a = batch['source'], batch['target']
        a = a.unsqueeze(1) 
        z0 = self.encode_images(images) 
        # update mine
        if optimizer_idx == 0:
            phi_z0 = self.perturb_net(z0).detach()
            mine_out = self.mine_net(z0, a)
            mine_out_phi = self.mine_net(phi_z0, a)
            mine_out_shuffled = self.mine_net(phi_z0, a[torch.randperm(a.size(0))])
            critic_loss = mine_out.mean() - torch.exp(mine_out_shuffled).mean()
            self.log('train/critic_loss', critic_loss, on_step=False, on_epoch=True, prog_bar=True)
            return -critic_loss  
        # update perturbation network
        elif optimizer_idx == 1:
            with torch.no_grad():
                mine_out_phi = self.mine_net(self.perturb_net(z0), a)
            perturb_loss = torch.exp(mine_out_phi).mean()
            self.log('train/perturb_loss', perturb_loss, on_step=False, on_epoch=True, prog_bar=True)
            return perturb_loss  
        

    def validation_step(self, batch, batch_idx):
        images, a = batch['source'], batch['target']
        a = a.unsqueeze(1)  
        z0 = self.encode_images(images)

        critic_loss = self.mine_loss(z0, a)
        perturb_loss = self.perturb_loss(z0, a)

        self.log('val/critic_loss', critic_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/perturb_loss', perturb_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_critic_loss": critic_loss, "val_perturb_loss": perturb_loss}

    def configure_optimizers(self):
        mine_optimizer = torch.optim.Adam(
            self.mine_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        perturb_optimizer = torch.optim.Adam(
            self.perturb_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [mine_optimizer, perturb_optimizer], []

    def configure_optimizers(self):
        mine_optimizer = Adam(
            self.mine_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        perturb_optimizer = Adam(
            self.perturb_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [mine_optimizer, perturb_optimizer], []