from __future__ import print_function
import os, math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from PIL import Image
from copy import deepcopy
import torchvision.models as models

attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows',
 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
 'Big_Lips', 'Big_Nose', 'Black_Hair',
 'Blond_Hair', 'Blurry', 'Brown_Hair',
 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
 'Young']

def set_random_seed(seed=999):
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

dataroot = "/projectnb/dl523/students/alanwyz/EC523-main/data/img_align_celeba/"

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, transform = None):
        self.transform = transform
        self.root = dataroot
        self.attr_txt = dataroot + 'list_attr_celeba.txt'
        self._parse()

    def _parse(self):
        self.im_paths = []
        self.ys = []

        def _to_binary(lst):
            return torch.tensor([0 if lab == '-1' else 1 for lab in lst])

        with open(self.attr_txt) as f:
            for line in f:
                assert len(line.strip().split()) == 41
                fl = line.strip().split()
                if fl[0][-4:] == '.jpg':
                    self.im_paths.append(self.root + fl[0])
                    self.ys.append(_to_binary(fl[1:]))

    def __len__(self):
        return len(self.ys)

    def mask(self, image, coords):
        x0, y0, x1, y1 = coords
        masked_image = image.clone()
        noise = torch.randn_like(masked_image[:, y0:y1, x0:x1])
        masked_image[:, y0:y1, x0:x1] = noise
        return masked_image

    def __getitem__(self, index):
        def img_load(index):
            imraw = Image.open(self.im_paths[index])
            im = self.transform(imraw)
            return im
        original_image = img_load(index)
        mask_coords = (52, 52, 76, 76)
        masked_image = self.mask(original_image, mask_coords)
        attributes = self.ys[index]
        return masked_image, original_image, attributes

def nonlinearity(x):
    return x*torch.sigmoid(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.proj = nn.Linear(emb_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        t_proj = self.proj(t)
        t_proj = nonlinearity(t_proj)
        t_proj = t_proj.view(t_proj.shape[0], t_proj.shape[1], 1, 1).expand(-1, -1, h.size(2), h.size(3))
        h = h + t_proj
        h = nonlinearity(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)
        return x+h

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = Block(in_channels, out_channels)

    def forward(self, x, t):
        x = self.pool(x)
        x = self.conv(x, t)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = Block(in_channels, out_channels)

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x, t)
        return x

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, conditional=True, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.inc = Block(c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.bot1 = Block(256, 512)
        self.bot2 = Block(512, 512)
        self.bot3 = Block(512, 512)
        self.bot4 = Block(512, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        self.conditional = conditional
        if conditional:
            num_classes = 2
            self.gender_vectors = nn.Parameter(torch.randn(num_classes, emb_dim))

    def temporal_encoding(self, timestep):
        assert len(timestep.shape) == 1
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timestep.device)
        emb = timestep.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0,1,0,0))
        return emb

    def unet_forward(self, x, t):
        x1 = self.inc(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x4 = self.bot1(x4, t)
        x4 = self.bot2(x4, t)
        x4 = self.bot3(x4, t)
        x4 = self.bot4(x4, t)
        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output

    def forward(self, x, t, y=None):
        if self.conditional:
            temp_emb = self.temporal_encoding(t)
            if y is None:
                y = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                y = y.to(device=x.device, dtype=torch.long)
            gender_emb = self.gender_vectors[y]
            c = temp_emb + gender_emb
        else:
            c = self.temporal_encoding(t)
        return self.unet_forward(x, c)

class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def get_noisy_image(self, x_0, t):
        B, C, H, W = x_0.shape
        x_0 = x_0.to(self.device)
        t = t.to(self.device)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        epsilon = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1. - alpha_bar_t) * epsilon
        return x_t, epsilon

    def sample(self, model, masked_images, timesteps, y=None):
        model.eval()
        with torch.no_grad():
            x_t = masked_images.to(self.device)
            if y is None:
                y = torch.randint(0, 2, (x_t.size(0),), dtype=torch.long, device=self.device)
            else:
                y = y.to(dtype=torch.long, device=self.device)
            for t in reversed(range(1, self.num_timesteps + 1)):
                beta_t = self.beta[t-1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                alpha_t = self.alpha[t-1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                alpha_bar_t = self.alpha_bar[t-1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                current_mask = (timesteps == t)
                if not current_mask.any():
                    continue
                current_x_t = x_t[current_mask]
                current_y = y[current_mask]
                t_tensor = torch.full((current_x_t.size(0),), t-1, dtype=torch.long, device=self.device)
                eps_theta = model(current_x_t, t_tensor, current_y)
                mu_theta = (current_x_t - (beta_t / torch.sqrt(1. - alpha_bar_t)) * eps_theta) / torch.sqrt(alpha_t)
                if t > 1:
                    sigma_t = torch.sqrt(beta_t * (1 - self.alpha_bar[t-2]) / (1 - self.alpha_bar[t-1]))
                    x_t[current_mask] = mu_theta + sigma_t * torch.randn_like(current_x_t)
                else:
                    x_t[current_mask] = mu_theta
        model.train()
        return (x_t.clamp(-1, 1) + 1) / 2

def show_images(images, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    if ndarr.dtype == np.float32:
        ndarr = np.clip(ndarr, 0, 1)
        ndarr = (255 * ndarr).astype(np.uint8)
    im = Image.fromarray(ndarr)
    plt.imshow(im)
    plt.show()

class EMA:
    def __init__(self, beta=0.995):
        super().__init__()
        self.beta = beta

    def step_ema(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        return old * self.beta + (1 - self.beta) * new

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:36].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.criterion = nn.MSELoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, input, target):
        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return self.criterion(input_features, target_features)

image_size = 128
batch_size = 64
learning_rate = 0.0001
weight_decay = 0.00001

train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = CelebADataset(transform=train_transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
gender_index = attributes.index('Male')
device = 'cuda'
model = UNet().to(device)
ema_model = deepcopy(model)
ema = EMA()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
diffusion = Diffusion(img_size=image_size, device=device)
scaler = torch.cuda.amp.GradScaler()
perceptual_loss = PerceptualLoss().to(device)
num_epoch = 20

for epoch in range(num_epoch):
    print(f'Epoch: {epoch+1}/{num_epoch}')
    pbar = tqdm(total=len(trainloader), desc="Training")
    for masked_images, original_images, attributes in trainloader:
        masked_images = masked_images.to(device)
        original_images = original_images.to(device)
        attributes = attributes.to(device)
        y = attributes[:, gender_index].long().to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            t = torch.randint(1, diffusion.num_timesteps, (masked_images.size(0),)).to(device)
            noisy_images, _ = diffusion.get_noisy_image(masked_images, t)
            predicted_noise = model(noisy_images, t, y)
            mse_loss = F.mse_loss(predicted_noise, noisy_images)
            percep_loss = perceptual_loss(noisy_images + predicted_noise, original_images)
            loss = mse_loss + percep_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.update(1)
        pbar.set_postfix(PerceptualLoss=loss.item(), LR=optimizer.param_groups[0]['lr'])
    if epoch == 0:
        ema_model = deepcopy(model)
    if epoch > 0:
        ema.step_ema(ema_model, model)
    set_random_seed()
    timesteps = torch.randint(1,diffusion.num_timesteps,(8,)).to(device)
    masked_images = masked_images[:8].to(device)
    sampled_images = diffusion.sample(ema_model, masked_images,timesteps, y=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).cuda())
    show_images(sampled_images)
    pbar.close()
