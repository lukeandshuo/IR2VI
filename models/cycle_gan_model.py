import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.roi_pooling import roi_pooling
from torchvision import transforms
import sys


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.object_size = (opt.objectSize,opt.objectSize)

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        self.input_A_bboxes = self.Tensor(nb,1,4)
        self.input_B_bboxes = self.Tensor(nb,1,4)
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            ###TODO change opt.which_model_netD  for define customized Discriminators
            self.netD_A_object = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B_object = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_A_object, 'D_A_object', which_epoch)
                self.load_network(self.netD_B_object, 'D_B_object', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.fake_A_object_pool = ImagePool(opt.pool_size)
            self.fake_B_object_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A_object = torch.optim.Adam(self.netD_A_object.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B_object = torch.optim.Adam(self.netD_B_object.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_A_object)
            self.optimizers.append(self.optimizer_D_B_object)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_A_object)
            networks.print_network(self.netD_B_object)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        if self.opt.isTrain:
            self.input_A_bboxes = input['A_bboxes' if AtoB else 'B_bboxes'].squeeze(1)
            self.input_B_bboxes = input['B_bboxes' if AtoB else 'A_bboxes'].squeeze(1)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        self.real_A_bboxes = Variable(self.input_A_bboxes)
        self.real_B_bboxes = Variable(self.input_B_bboxes)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

        ## for object
        fake_B_object = self.fake_B_object_pool.query(self.fake_B_object)
        real_B_object = roi_pooling(self.real_B,self.real_B_bboxes,size=self.object_size)
        loss_D_A_object = self.backward_D_basic(self.netD_A_object, real_B_object, fake_B_object)
        self.loss_D_A_object = loss_D_A_object.data[0]

        # Combine loss
        loss_D_A_total = (5*loss_D_A + loss_D_A_object)
        # backward
        loss_D_A_total.backward()

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]

        ## for object
        fake_A_object = self.fake_A_object_pool.query(self.fake_A_object)
        real_A_object = roi_pooling(self.real_A,self.real_A_bboxes,size=self.object_size)
        loss_D_B_object = self.backward_D_basic(self.netD_B_object, real_A_object, fake_A_object)
        self.loss_D_B_object = loss_D_B_object.data[0]

        # Combine loss
        loss_D_B_total = (5 * loss_D_B + loss_D_B_object)
        # backward
        loss_D_B_total.backward()

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_A_object(G_A(A))
        fake_B_object = roi_pooling(fake_B,self.real_A_bboxes,size=self.object_size)
        pred_fake_object = self.netD_B_object(fake_B_object)
        loss_G_A_object = self.criterionGAN(pred_fake_object,True)
        ###TODO: Debug
        # test = transforms.ToPILImage()(fake_B_object.cpu().data.squeeze(0))
        # test.show()

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # GAN loss D_A_object(G_A(A))
        fake_A_object = roi_pooling(fake_A,self.real_B_bboxes,size=self.object_size)
        pred_fake_object = self.netD_A_object(fake_A_object)
        loss_G_B_object = self.criterionGAN(pred_fake_object,True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A
        # Forward cycle object loss
        rec_A_object = roi_pooling(rec_A,self.real_A_bboxes,size=self.object_size)
        real_A_object = roi_pooling(self.real_A,self.real_A_bboxes,size=self.object_size)
        loss_cycle_A_object = self.criterionCycle(rec_A_object, real_A_object) * lambda_A

        # Backward cycle loss
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B
        # Backward cycle object loss
        rec_B_object = roi_pooling(rec_B,self.real_B_bboxes,size=self.object_size)
        real_B_object = roi_pooling(self.real_B,self.real_B_bboxes,size=self.object_size)
        loss_cycle_B_object = self.criterionCycle(rec_B_object, real_B_object) * lambda_B

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + 0.1*(loss_G_A_object + loss_G_B_object) + 0.1*(loss_cycle_A_object+ loss_cycle_B_object)
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data
        self.real_A_object = real_A_object.data
        self.real_B_object = real_B_object.data
        self.fake_B_object = fake_B_object.data
        self.fake_A_object = fake_A_object.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_G_A_object = loss_G_A_object.data[0]
        self.loss_G_B_object = loss_G_B_object.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        self.loss_cycle_A_object = loss_cycle_A_object.data[0]
        self.loss_cycle_B_object = loss_cycle_B_object.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B),
                                  ('D_A_object',self.loss_D_A_object),('G_A_object',self.loss_G_A_object),('Cyc_A_object', self.loss_cycle_A_object),
                                  ('D_B_object',self.loss_D_B_object),('G_B_object',self.loss_G_B_object),('Cyc_B_object',  self.loss_cycle_B_object),])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)


        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        if self.opt.isTrain:
            ret_visuals['fake_A_object'] = util.tensor2im(self.fake_A_object)
            ret_visuals['fake_B_object'] = util.tensor2im(self.fake_B_object)
            ret_visuals['real_B_object'] = util.tensor2im(self.real_B_object)
            ret_visuals['real_A_object'] = util.tensor2im(self.real_A_object)

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

        self.save_network(self.netD_B_object, 'D_A_object', label, self.gpu_ids)
        self.save_network(self.netD_B_object, 'D_B_object', label, self.gpu_ids)
