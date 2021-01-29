import argparse

from implementations.data_utils import load_data
from implementations.visualize import plot_losses
from implementations.torch_utils import to_var, calc_gradient_penalty
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500, help="number of epochs of training")
parser.add_argument("--hidden", type=int, default=512, help="number of neurons in hidden layer")
parser.add_argument("--batch", type=int, default=64, help="number of batch size")
parser.add_argument("--show_loss", type=int, default=100, help="number of epochs of showing loss")
parser.add_argument("--d_steps", type=int, default=10, help="number of epochs of showing loss")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")
parser.add_argument("--sample_dir", type=str, default="./figures/", help="binary or multi for discriminator classification task")
parser.add_argument("--classification", type=str, default="binary", help="binary or multi for discriminator classification task")
parser.add_argument("--generator_model", type=str, default="Gen_Lin", help="choose discriminator model")
parser.add_argument("--discriminator_model", type=str, default="Dis_Lin", help="choose discriminator model")
parser.add_argument("--loss", type=str, default="", help="choose loss")
parser.add_argument("--optimizer", type=str, default="Adam", help="choose optimizer")

opt = parser.parse_args()
classification = opt.classification 
generator_model = opt.generator_model
discriminator_model = opt.discriminator_model
optimizer = opt.optimizer
sample_dir = opt.sample_dir
use_cuda  = True if torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda  else "cpu")

# def prepare_data_model():
dataset, pos_amino, virus_nparr, max_len, amino_num = load_data(classification) #numpy.ndarray
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
print(dataloader, pos_amino, virus_nparr, max_len, amino_num)

if classification == "multi":
    out_dim = len(y[0])
if classification == "binary":
    out_dim = 2

if generator_model == "Gen_Lin_Block_CNN":
    G = Gen_Lin_Block_CNN(max_len, amino_num, out_dim, opt.hidden)
if generator_model == "Gen_Lin":
    G = Gen_Lin(max_len, amino_num, out_dim, opt.hidden)
if discriminator_model == "Dis_Lin_Block_CNN":
    D = Dis_Lin_Block_CNN(max_len, amino_num, out_dim, opt.hidden)
if discriminator_model == "Dis_Lin":
    D = Dis_Lin(max_len, amino_num, out_dim, opt.hidden)

if use_cuda:
    G = G.cuda()
    D = D.cuda()

# print(G)
# print(D)

if optimizer == "Adam":
    G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.9))


def train_model():
    d_fake_losses, d_real_losses, grad_penalties = [],[],[]
    G_losses, D_losses, W_dist = [],[],[]

    one = torch.tensor(1, dtype=torch.float)
    one = one.cuda() if use_cuda else one
    one_neg = one * -1

    for epoch in range(opt.epoch):
        for i, (data, _) in enumerate(dataloader):
            real_data = data[0].to(device).float()   
            real_data = to_var(real_data)         

            for p in D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for _ in range(opt.d_steps): # Train D
                D.zero_grad()
                d_real_pred = D(real_data)
                d_real_err = torch.mean(d_real_pred) #want to push d_real as high as possible
                d_real_err.backward(one_neg)

                z_input = torch.randn(opt.batch, max_len*amino_num)
                z_input = z_input.to(device).float()
                z_input = to_var(z_input)

                d_fake_data = G(z_input).detach()
                d_fake_pred = D(d_fake_data)
                d_fake_err = torch.mean(d_fake_pred) #want to push d_fake as low as possible
                d_fake_err.backward(one)

                gradient_penalty = calc_gradient_penalty(real_data.data, d_fake_data.data, opt.batch, D)
                gradient_penalty.backward()

                d_err = d_fake_err - d_real_err + gradient_penalty
                D_optimizer.step()

            for p in D.parameters():
                p.requires_grad = False  # to avoid computation

            G.zero_grad()
            z_input = to_var(torch.randn(opt.batch, max_len*amino_num))
            g_fake_data = G(z_input)
            dg_fake_pred = D(g_fake_data)
            g_err = -torch.mean(dg_fake_pred)
            g_err.backward()
            G_optimizer.step()


            # Append things for logging
            d_fake_np, d_real_np, gp_np = (d_fake_err.data).cpu().numpy(), \
                    (d_real_err.data).cpu().numpy(), (gradient_penalty.data).cpu().numpy()
            grad_penalties.append(gp_np)
            d_real_losses.append(d_real_np)
            d_fake_losses.append(d_fake_np)
            D_losses.append(d_fake_np - d_real_np + gp_np) # minus(real - fake)
            G_losses.append((g_err.data).cpu().numpy())
            W_dist.append(d_real_np - d_fake_np)

            if i % 5 == 0:
                summary_str = 'Iteration {} - loss_d: {}, loss_g: {}, w_dist: {}, grad_penalty: {}'\
                    .format(i, (d_err.data).cpu().numpy(),
                    (g_err.data).cpu().numpy(), ((d_real_err - d_fake_err).data).cpu().numpy(), gp_np)
                print(summary_str)
                plot_losses([G_losses, D_losses], ["gen", "disc"], sample_dir + "losses.png")
                plot_losses([W_dist], ["w_dist"], sample_dir + "dist.png")
                plot_losses([grad_penalties],["grad_penalties"], sample_dir + "grad.png")
                plot_losses([d_fake_losses, d_real_losses],["d_fake", "d_real"], sample_dir + "d_loss_components.png")


def main():
    train_model()

if __name__ == '__main__':
    main()
