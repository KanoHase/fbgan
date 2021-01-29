from transformers import AutoTokenizer, AutoModel

import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from implementations.data_utils import load_data

# os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--lambda_gp", type=float, default=10, help="Loss weight for gradient penalty")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--hidden", type=int, default=512, help="dimensionality of the hidden layer")
parser.add_argument("--seq_len", type=int, default=60, help="size of amino acid sequence")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=10, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.seq_len, opt.seq_len)

cuda = True if torch.cuda.is_available() else False
n_chars = 20

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(opt.latent_dim, opt.hidden*opt.seq_len)
        self.block = nn.Sequential(
            ResBlock(opt.hidden),
            ResBlock(opt.hidden),
            ResBlock(opt.hidden),
            ResBlock(opt.hidden),
            ResBlock(opt.hidden),
        )
        self.conv1 = nn.Conv1d(opt.hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = opt.seq_len
        self.batch_size = opt.batch_size
        self.hidden = opt.hidden

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len, -1)
        output = gumbel_softmax(output, 0.5)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.n_chars = n_chars
        self.seq_len = opt.seq_len
        self.batch_size = opt.batch_size
        self.hidden = opt.hidden
        self.block = nn.Sequential(
            ResBlock(opt.hidden),
            ResBlock(opt.hidden),
            ResBlock(opt.hidden),
            ResBlock(opt.hidden),
            ResBlock(opt.hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, opt.hidden, 1)
        self.linear = nn.Linear(opt.seq_len*opt.hidden, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len*self.hidden)
        output = self.linear(output)
        return output

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
"""
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.seq_len), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
"""
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (seqs, _) in enumerate(dataloader):

        # Configure input
        real_seqs = Variable(seqs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (seqs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_seqs = generator(z)

        # Real images
        real_validity = discriminator(real_seqs)
        # Fake images
        fake_validity = discriminator(fake_seqs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_seqs.data, fake_seqs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_seqs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_seqs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                save_image(fake_seqs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic