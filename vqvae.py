 #!/usr/bin/env python -W ignore::DeprecationWarning
'''
Credit: https://github.com/ritheshkumar95/pytorch-vqvae
'''

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import torch.nn as nn
from Audio_vqvae import audio_vqvae
from torch.nn import DataParallel

from vq_modules import VectorQuantizedVAE, to_scalar
from preproc import DataVAE, collate_vae

from tensorboardX import SummaryWriter

def train(data_loader, model, optimizer, args, writer, file_):
    for batch in data_loader:
        feats = batch.to(args.device)
        #feats = feats.unsqueeze(2)
        #print("FEATS:", feats.shape)

        optimizer.zero_grad()
        x_tilde, vq_loss, losses, perplexity, \
            encoding_indices, concatenated_quantized = model(feats)
        pred_pad = nn.ZeroPad2d(padding=(0, feats.shape[2]-x_tilde.shape[2], 
                                feats.shape[1]-x_tilde.shape[1], 0))
        x_tilde = pred_pad(x_tilde)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, feats)
        # Vector quantization objective
        #loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        #loss_commit = F.mse_loss(z_ex, z_q_x.detach())

        loss = loss_recons + vq_loss #+ args.beta * loss_commit
        loss.backward()
        print("Training loss:", loss.detach().cpu().numpy())

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        #writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)
        
        optimizer.step()
        args.steps += 1
        f = open(args.logs+'results.txt', 'w+')
        f.write("step "+str(args.steps)+":"+str(loss.detach().cpu().numpy())+'\n')
        f.close()

def test(data_loader, model, args, writer, file_):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for batch in data_loader:
            feats = batch.to(args.device)
            x_tilde, vq_loss, losses, perplexity, \
            encoding_indices, concatenated_quantized = model(feats)
            pred_pad = nn.ZeroPad2d(padding=(0, feats.shape[2]-x_tilde.shape[2], 
                                    feats.shape[1]-x_tilde.shape[1], 0))
            x_tilde = pred_pad(x_tilde)
            loss_recons += F.mse_loss(x_tilde, feats)
            #print(loss_recons)
            loss_vq += vq_loss

        loss_recons /= len(data_loader)
        vq_loss /= len(data_loader)
        print("Validation Loss:", loss_recons.detach().cpu().numpy() + vq_loss.detach().cpu().numpy())
        f = open(args.logs+'results.txt', 'w+')
        f.write("Validation:"+str(loss_recons.detach().cpu().numpy() + vq_loss.detach().cpu().numpy())+'\n')
        f.close()
    # Logs
    #writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    #writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)
    return loss_recons, vq_loss
    #return loss_recons.item(), loss_vq.item()

def generate_samples(feats, model, args):
    with torch.no_grad():
        feats = feats.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde

def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}'.format(args.output_folder)
    
    train_dataset = DataVAE('/nobackup/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/train.tsv',
                         '/nobackup/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/clips/')

    valid_dataset = DataVAE('/nobackup/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/dev.tsv',
                        '/nobackup/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/clips/')
    test_dataset = DataVAE('/nobackup/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/test.tsv',
                         '/nobackup/ak16/Basque/cv-corpus-5.1-2020-06-22/eu/clips/')

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_vae)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_vae)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True, collate_fn=collate_vae)

    # Fixed images for Tensorboard

    model = audio_vqvae(input_dim=39, 
                        hid_dim=args.hidden_size, 
                        enc_dim=64, 
                        K=args.k).to(args.device)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    #model = VectorQuantizedVAE(39, args.hidden_size, args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    best_loss = -1
    
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, writer, f)
        loss, _ = test(valid_loader, model, args, writer,f)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')

    parser.add_argument('--logs', type=str , default='logs/')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)