import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse
from utils.torch_utils import *
from utils.utils import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np
from models import *
import random

class WGAN_LangGP():
    def __init__(self, batch_size=64, lr=0.0001, num_epochs=400, seq_len = 156, data_dir='./data/dna_uniprot_under_50_reviewed.fasta', \
        run_name='test', hidden=512, d_steps = 10, max_examples=2000, divide_epochs = 4):
        # self.preds_cutoff = 0.8
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = 1
        self.lamda = 10 #lambda
        self.divide_epochs = divide_epochs #must be 3 or above
        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        self.sample_dir = './samples/' + run_name + "/"
        self.load_data(data_dir, max_examples) #max examples is size of discriminator training set
        self.load_real_dna('./data/dna_positive_544.txt', max_examples)
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()

    def build_model(self):
        self.G = Generator_FBGAN(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        self.D = Discriminator_FBGAN(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda() 
        print(self.G)
        print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        # val_loss, val_acc = self.analyzer.evaluate_model()
        # print("Val Acc:{}".format(val_acc))

    def load_data(self, datadir, max_examples=1e6):#taken data from './data/dna_uniprot_under_50_reviewed.fasta'
        self.data, self.charmap, self.inv_charmap = utils.language_helpers.load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        ) #self.data is the list of all data(DNA) : ('A', 'T', 'G', ...)
        #self.charmap:{'P': 0, 'A': 1, 'G': 2, 'T': 3, 'C': 4}, self.inv_charmap:['P', 'A', 'G', 'T', 'C'], len(self.data))==2000(もとは3655このデータ)
        self.labels = np.zeros(len(self.data)) #this marks at which epoch this data was added
        print("!!!!!!!!!",self.data[0])
    
    def load_real_dna(self, datadir, max_examples=1e6):
        self.real_data, self.real_charmap, self.real_inv_charmap = utils.language_helpers.load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        ) 
        random.shuffle(self.real_data) #shuffle
        # self.real_labels = np.zeros(len(self.real_data)) 

    def remove_old_indices(self, numToAdd):
        toRemove = np.argsort(self.labels)[:numToAdd] #toRemove is the list of index that is to be removed
        self.data = [d for i,d in enumerate(self.data) if i not in toRemove]
        self.labels = np.delete(self.labels, toRemove)

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), self.checkpoint_dir + "G_weights_{}.pth".format(epoch))
        torch.save(self.D.state_dict(), self.checkpoint_dir + "D_weights_{}.pth".format(epoch))

    def load_model(self, directory = '', iteration=None):
        '''
            Load model parameters from most recent epoch
        '''
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth") #list of the files' names 
        list_D = glob.glob(directory + "D*.pth")
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        if iteration is None:
            print("Loading most recently saved...")
            G_file = max(list_G, key=os.path.getctime)
            D_file = max(list_D, key=os.path.getctime)
        else:
            G_file = "G_weights_{}.pth".format(iteration)
            D_file = "D_weights_{}.pth".format(iteration)
        epoch_found = int( (G_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_found, directory))
        self.G.load_state_dict(torch.load(G_file,map_location='cpu'))
        self.D.load_state_dict(torch.load(D_file,map_location='cpu'))
        return epoch_found

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1)
        alpha = alpha.view(-1,1,1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda() if self.use_cuda else alpha
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda() if self.use_cuda else interpolates
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                                  if self.use_cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1).norm(2,dim=1) - 1) ** 2).mean() * self.lamda
        return gradient_penalty

    def train_model(self, load_dir):
        self.load_model(load_dir)
        losses_f = open(self.checkpoint_dir + "losses.txt",'a+')
        d_fake_losses, d_real_losses, grad_penalties = [],[],[]
        G_losses, D_losses, W_dist = [],[],[]

        # one = torch.FloatTensor([1]):this doesn't work for CPU
        one = torch.tensor(1, dtype=torch.float)
        one = one.cuda() if self.use_cuda else one
        one_neg = one * -1 #tensor([-1.])
        table = np.arange(len(self.charmap)).reshape(-1, 1)
        print("=========",table)
        one_hot = OneHotEncoder()
        one_hot.fit(table)
        num_batches_sample = 15
        n_batches = int(len(self.data)/self.batch_size)

        i = 0
        if len(self.real_data)>=len(self.data):
            ip_batches = math.ceil(len(self.real_data)/(self.n_epochs*2/self.divide_epochs))
        else:
            ip_batches = math.ceil(len(self.real_data)/(self.n_epochs/self.divide_epochs))
        #ip_batches: batch for real positive peptides
        
        cutoff_epochs = int(self.n_epochs/self.divide_epochs)
        shrink_size = int((len(self.data) - len(self.real_data)) / cutoff_epochs)
        #cutoff_epochs: epochs to cutoff random peptides
        #shrink_size: how many random peptides to cut per epoch

        print(ip_batches, cutoff_epochs, shrink_size)        

        for epoch in range(1, self.n_epochs+1):
            if epoch % 2 == 0: self.save_model(epoch)
            sampled_seqs = self.sample(num_batches_sample, epoch) #letters

            with open(self.sample_dir + "sampled_{}_preds.txt".format(epoch), 'w+') as f:
                f.writelines([s + '\n' for s in sampled_seqs])#[s + '\t' + str(preds[j][0]) + '\n' for j, s in enumerate(valid_gene_seqs)])
            
            if (epoch-1)*ip_batches+1<= len(self.real_data):
                #print((epoch-1)*ip_batches, epoch*ip_batches)
                if epoch*ip_batches <= len(self.real_data):
                    pos_seqs = [list(self.real_data[k]) for k in range((epoch-1)*ip_batches, epoch*ip_batches)]
                else:
                    pos_seqs = [list(self.real_data[k]) for k in range((epoch-1)*ip_batches, len(self.real_data))]
                self.remove_old_indices(len(pos_seqs))
                self.data += pos_seqs
                #print("========",pos_seqs)
            else:
                #print(epoch)
                count_zeros = list(self.labels).count(0)
                #print(count_zeros)

                if count_zeros < shrink_size:
                    self.remove_old_indices(count_zeros)
                    n_batches = int(len(self.data)/self.batch_size)
                    #print("@@@@@@@@@@@",len(self.data), '\t', n_batches)

                else:
                    self.remove_old_indices(shrink_size)
                    n_batches = int(len(self.data)/self.batch_size)
                    #print("!!!!!!!!!!!!!!",len(self.data), '\t', n_batches)

            self.labels = np.concatenate([self.labels, np.repeat(epoch, len(pos_seqs))] )
            #print("---------------",self.labels, len(self.labels))
            perm = np.random.permutation(len(self.data))
            self.data = [self.data[i] for i in perm]
            self.labels = self.labels[perm]

            for idx in range(n_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in self.data[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                print("----------",data_one_hot, data_one_hot.shape)
                real_data = torch.Tensor(data_one_hot)
                real_data = to_var(real_data)
                for p in self.D.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                for _ in range(self.d_steps): # Train D
                    self.D.zero_grad()
                    print("********",type(real_data))
                    d_real_pred = self.D(real_data)
                    d_real_err = torch.mean(d_real_pred) #want to push d_real as high as possible  # d_real_err:tensor(4.9055, grad_fn=<MeanBackward0>)
                    d_real_err.backward(one_neg)
                    z_input = to_var(torch.randn(self.batch_size, 128))
                    d_fake_data = self.G(z_input).detach()
                    d_fake_pred = self.D(d_fake_data)
                    d_fake_err = torch.mean(d_fake_pred) #want to push d_fake as low as possible
                    d_fake_err.backward(one)

                    gradient_penalty = self.calc_gradient_penalty(real_data.data, d_fake_data.data)

                    gradient_penalty.backward()

                    d_err = d_fake_err - d_real_err + gradient_penalty
                    self.D_optimizer.step()

                # Append things for logging
                d_fake_np, d_real_np, gp_np = (d_fake_err.data).cpu().numpy(), \
                        (d_real_err.data).cpu().numpy(), (gradient_penalty.data).cpu().numpy()
                grad_penalties.append(gp_np)
                d_real_losses.append(d_real_np)
                d_fake_losses.append(d_fake_np)
                D_losses.append(d_fake_np - d_real_np + gp_np) # minus(real - fake)
                W_dist.append(d_real_np - d_fake_np)
                # Train G
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                z_input = to_var(torch.randn(self.batch_size, 128))
                g_fake_data = self.G(z_input)
                dg_fake_pred = self.D(g_fake_data)
                g_err = -torch.mean(dg_fake_pred)
                g_err.backward()
                self.G_optimizer.step()
                G_losses.append((g_err.data).cpu().numpy())
                if i % 10 == 9:
                    summary_str = 'Iteration {} - loss_d: {}, loss_g: {}, w_dist: {}, grad_penalty: {}'\
                        .format(i, (d_err.data).cpu().numpy(),
                        (g_err.data).cpu().numpy(), ((d_real_err - d_fake_err).data).cpu().numpy(), gp_np)
                    print(summary_str)
                    losses_f.write(summary_str)
                    plot_losses([G_losses, D_losses], ["gen", "disc"], self.sample_dir + "losses.png")
                    plot_losses([W_dist], ["w_dist"], self.sample_dir + "dist.png")
                    plot_losses([grad_penalties],["grad_penalties"], self.sample_dir + "grad.png")
                    plot_losses([d_fake_losses, d_real_losses],["d_fake", "d_real"], self.sample_dir + "d_loss_components.png")
                i += 1



    def sample(self, num_batches_sample, epoch):
        decoded_seqs = []
        for i in range(num_batches_sample):
            z = to_var(torch.randn(self.batch_size, 128))#torch.utilsにある、Variable(x)を返す
            self.G.eval()
            torch_seqs = self.G(z) #torch
            seqs = (torch_seqs.data).cpu().numpy()
            decoded_seqs += [decode_one_seq(seq, self.inv_charmap) for seq in seqs]#len(decoded_seqs):960(64*15)
            # decoded_seqs += [decode_one_seq(seq, ['P', 'A', 'T', 'G', 'C']) for seq in seqs]
        self.G.train()
        return decoded_seqs

def main():
    parser = argparse.ArgumentParser(description='FBGAN with AVP analyzer.')
    parser.add_argument("--run_name", default= "fbgan_avp_demo", help="Name for output files")
    parser.add_argument("--load_dir", default="./checkpoint/realProt_50aa/", help="Load pretrained GAN checkpoints")
    args = parser.parse_args()
    model = WGAN_LangGP(run_name=args.run_name)
    model.train_model(args.load_dir)

if __name__ == '__main__':
    main()
