import os

import torch
import torch.optim as optim 
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 
from torchvision.utils import save_image 
from net import Discriminator, Generator


def train(D, G, loader, epoch, lsgan_params, batch_size=128, z_dim=62, device='cpu'):

    # optimizer
    D_optim = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))

    # hyper parameter
    a, b, c = lsgan_params

    for i in range(epoch):

        y_real = torch.ones(batch_size, 1)
        y_fake = torch.zeros(batch_size, 1)
    
        D_running_loss = 0 
        G_running_loss = 0 

        for batch_index, (real_img, _) in enumerate(loader):
            if real_img.size()[0] != batch_size:
                break 

            real_img = real_img.to(device)

            # random sampling from latent space 
            z = torch.rand(batch_size, z_dim)
            z = z.to(device)

            ### Update Discriminator 
            D_optim.zero_grad() 

            # real 
            D_real = D(real_img)
            D_real_loss = torch.sum((D_real - b) ** 2)

            # fake 
            fake_img = G(z)
            D_fake = D(fake_img.detach()) # stop back propagation to G 
            D_fake_loss = torch.sum((D_fake - a) ** 2)

            # minimizing loss 
            D_loss = 0.5 * (D_real_loss + D_fake_loss) / batch_size
            D_loss.backward()
            D_optim.step()
            D_running_loss += D_loss.data.item()

            ### Update Generator 
            G_optim.zero_grad()

            fake_img = G(z)
            D_fake = D(fake_img)

            G_loss = 0.5 * (torch.sum((D_fake - c) ** 2)) / batch_size 
            G_loss.backward()
            G_optim.step()
            G_running_loss += G_loss.data.item()
        
        print('epoch: {:d} loss_d: {:.3f} loss_g: {:.3f}'.format(i+1, D_running_loss/batch_size, G_running_loss/batch_size))

        # save image
        path = 'result/mnist'
        if not os.path.exists(path):
            os.makedirs(path)

        save_image(fake_img, '{}/image{:d}.png'.format(path, i+1), nrow=16, normalize=True)

    torch.save(G.state_dict(), '{}/g.pth'.format(path))
    torch.save(D.state_dict(), '{}/d.pth'.format(path))


if __name__ == '__main__':
    
    # dataset 
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # args 
    z_dim = 62 
    epoch = 50
    batch_size = 128
    lsgan_params = [0, 1, 1]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize modules 
    D = Discriminator()
    G = Generator()

    train(D.to(device), G.to(device), loader, epoch, lsgan_params, batch_size, z_dim, device)
