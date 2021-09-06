import argparse
import torch
import itertools
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision.utils import save_image

from cycleGAN import Gen, Dis
from dataset import CustomDataset, weights_init
from utils import *
from loss import real_target_loss, fake_target_loss, cycle_loss, identity_loss

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str)
parser.add_argument('--target', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--val',type=str, default=None)

args = parser.parse_args()

def train():
    gen_X2Y = Gen().apply(weights_init).cuda(args.gpu)
    gen_Y2X = Gen().apply(weights_init).cuda(args.gpu)
    dis_X = Dis().apply(weights_init).cuda(args.gpu)
    dis_Y = Dis().apply(weights_init).cuda(args.gpu)

    learning_rate = 0.0002
    batch_size = 1
    optimizer_G = torch.optim.Adam(itertools.chain(gen_X2Y.parameters(), gen_Y2X.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_X = torch.optim.Adam(dis_X.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_Y = torch.optim.Adam(dis_Y.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    dataset = CustomDataset(args.input, args.target, args.image_size)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # target_loss = nn.BCELoss().cuda(args.gpu)
    # identity_loss = nn.L1Loss().cuda(args.gpu)
    # cycle_loss = nn.L1Loss().cuda(args.gpu)
    target_loss = nn.MSELoss()
    identity_loss = nn.L1Loss()
    cycle_loss = nn.L1Loss()
    best_gen_loss = 99999999.0

    for epoch in tqdm(range(1, args.epochs+1)):
        gen_X2Y.train()
        gen_Y2X.train()
        dis_X.train()
        dis_Y.train()
        total_gen_loss = 0.0
        total_dis_X_loss = 0.0
        total_dis_Y_loss = 0.0
        for i, (X, Y) in enumerate(train_loader):
            optimizer_G.zero_grad()

            X, Y = X.cuda(args.gpu), Y.cuda(args.gpu)
            fake_X, fake_Y = gen_Y2X(Y), gen_X2Y(X)
            re_X, re_Y = gen_Y2X(fake_Y), gen_X2Y(fake_X)
            identity_X, identity_Y = gen_Y2X(X), gen_X2Y(Y)
            dis_f_X, dis_f_Y = dis_X(fake_X), dis_Y(fake_Y)

            Adv_loss_X = target_loss(dis_f_X, torch.ones_like(dis_f_X))
            Adv_loss_Y = target_loss(dis_f_Y, torch.ones_like(dis_f_Y))
            identity_loss_X = identity_loss(identity_X, X)
            identity_loss_Y = identity_loss(identity_Y, Y)
            cycle_loss_X = cycle_loss(X, re_X)
            cycle_loss_Y = cycle_loss(Y, re_Y)

            loss_G = Adv_loss_X + Adv_loss_Y + (identity_loss_X + identity_loss_Y) * 5.0 + (cycle_loss_X + cycle_loss_Y) * 10.0

            loss_G.backward()
            optimizer_G.step()

            dis_fake_X = dis_X(fake_X.detach())
            dis_real_X = dis_X(X)
            dis_fake_Y = dis_Y(fake_Y.detach())
            dis_real_Y = dis_Y(Y)

            loss_D_X = (target_loss(dis_fake_X, torch.zeros_like(dis_fake_X)) + target_loss(dis_real_X, torch.ones_like(dis_real_X))) * 0.5
            loss_D_Y = (target_loss(dis_fake_Y, torch.zeros_like(dis_fake_Y)) + target_loss(dis_real_Y, torch.ones_like(dis_real_Y))) * 0.5

            optimizer_D_X.zero_grad()
            loss_D_X.backward()
            optimizer_D_X.step()

            optimizer_D_Y.zero_grad()
            loss_D_Y.backward()
            optimizer_D_Y.step()

            total_gen_loss += loss_G.item()
            total_dis_X_loss += loss_D_X.item()
            total_dis_Y_loss += loss_D_Y.item()
            if i % 100 == 0:
                print('iter {:} | loss = {:}, {:}, {:}'.format(i,total_gen_loss/(i+1),total_dis_X_loss/(i+1),total_dis_Y_loss/(i+1)))


        if best_gen_loss > total_gen_loss:
            best_gen_loss = total_gen_loss
            torch.save(gen_X2Y.state_dict(),'ckpt/gen_X2Y.pth')
            torch.save(gen_Y2X.state_dict(),'ckpt/gen_Y2X.pth')
            torch.save(dis_X.state_dict(),'ckpt/dis_X.pth')
            torch.save(dis_Y.state_dict(),'ckpt/dis_Y.pth')
            print('Epoch : {:} | gen loss : {:} | dis X loss : {:} | dis Y loss : {:}'.format(epoch,
                                                                                          total_gen_loss/len(train_loader),
                                                                                          total_dis_X_loss/len(train_loader),
                                                                                          total_dis_Y_loss/len(train_loader)))

def val():
    gen_X2Y = Gen().cuda(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    gen_X2Y.load_state_dict(torch.load(f'{args.val}/gen_X2Y.pth',device))
    gen_X2Y.eval()

    input_images = CustomDataset(args.input, args.target, args.image_size, mode='val')
    l = input_images.__len__()

    for i in tqdm(range(l)):
        input_image, _ = input_images.__getitem__(i)
        input_image = input_image.unsqueeze(0).cuda(args.gpu)
        input_name = input_images.get_file_name(i)
        output = gen_X2Y(input_image)
        output = output.to(torch.device('cpu'))
        output = unnormalize(output)

        save_image(output, f'result/{input_name}')


def main():
    if args.val is None:
        train()
    else:
        val()

if __name__ == '__main__':
    main()




