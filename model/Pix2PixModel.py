import torch
from . import networks
import options as opt
from .BaseModel import BaseModel
import torch.nn.functional as F


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, g, d):

        opt.beta1 = 0.5
        opt.lambda_L1 = 10
        opt.lambda_SMOOTH_LOSS = 1
        opt.gan_mode = 'lsgan'
        opt.device = 'cuda'

        self.opt = opt
        self.device = opt.device
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.isTrain = True
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # self.netG = networks.define_G(3, 1, 64, 'resnet_9blocks', 'instance',
        #                               not True, 'normal', 0.02).to(self.device)
        self.netG = g  # g.to(self.device)
        self.netD = d
        self.loss_G = -1
        self.loss_D = -1
        self.loss_G_L1 = -1
        self.loss_G_GAN = -1
        self.loss_G_grad = -1
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
            # self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

    def set_input(self, source, target):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = torch.cat((source[:, :3, :, :], target[:, :3, :, :]), dim=0).to(self.device)
        self.real_B = torch.cat((source[:, 5, None, :, :], target[:, 5, None, :, :]), dim=0).to(self.device)
        self.depth_mask = self.real_B > 0
        self.depth_scaler = self.real_B.view(self.real_B.shape[0], -1).max(dim=1)[0]
        # self.depth_shift = self.real_B.view(self.real_B.shape[0], -1).min(dim=1)[0]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)[('depth', -1, -1)]  # G(A)
        self.fake_B = self.fake_B * self.depth_mask
        # self.fake_B = self.fake_B * self.depth_scaler.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # SCALE = 36
        # MIN_DEPTH = 1e-3
        # # MAX_DEPTH = 5
        # SCALE = 36  # we set baseline=0.0015m which is 36 times smaller than the actual value (0.54m)
        #
        # # MIN_DEPTH  = 1e-3
        # MAX_DEPTH = 6#self.depth_scaler.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # # max_disp = torch.tensor(1 / MIN_DEPTH)
        # # min_disp = 1 / MAX_DEPTH
        # self.fake_B = self.netG(self.real_A)[('depth', -1, -1)]  # G(A)
        # # min_disp = 1 / MAX_DEPTH
        # # max_disp = 1 / MIN_DEPTH
        # # depth = 1 / (self.fake_B.cpu().numpy() * max_disp + min_disp) * SCALE
        # # # self.fake_B = 1 / (self.fake_B * max_disp + min_disp) * SCALE
        # # self.fake_B = self.fake_B  * MAX_DEPTH
        #
        # min_disp = 1 / MAX_DEPTH  # 0.01
        # max_disp = 1 / MIN_DEPTH  # 10
        # scaled_disp = min_disp + (max_disp - min_disp) * self.fake_B  # (10-0.01)*disp+0.01
        # self.fake_B = 1 / scaled_disp
        # self.fake_B = self.fake_B * self.depth_mask


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = 4e-3 * (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self,add_g_loss):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 2e-3
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_grad = self.get_smooth_loss(self.fake_B, self.real_B) * self.opt.lambda_SMOOTH_LOSS
        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN * 2e-3 + self.loss_G_L1
        if add_g_loss:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_grad
        else:
            self.loss_G = self.loss_G_L1 + self.loss_G_grad
        self.loss_G.backward()

    def test(self):
        self.forward()  # compute fake images: G(A)
        return self.fake_B[:opt.batch_size, ...], self.fake_B[opt.batch_size:, ...]

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        a1 = 0.5
        a2 = 0.5
        # img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-a1 * img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-a2 * img_dyy.abs().mean(1, True)))

        return smooth1+smooth2
    def optimize_parameters(self, epoch=10):
        iteration = 3
        for i in range(iteration):  # discriminator-generator balancing

            self.forward()  # compute fake images: G(A)
            if i == iteration - 1 and epoch > 3:
                # update D
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.optimizer_D.zero_grad()  # set D's gradients to zero
                self.backward_D()  # calculate gradients for D
                self.optimizer_D.step()  # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G(epoch > 3)  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights
        return self.fake_B[:opt.batch_size, ...], self.fake_B[opt.batch_size:, ...]
