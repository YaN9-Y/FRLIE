import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import Encoder, Decoder, FeatureProcessNet
from .loss import  SSIM, TVLoss, HueLoss


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        if config.MODEL == 1:
            self.name = 'reconstruct'
        elif config.MODEL == 2:
            self.name = 'feature_process'



        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, 'weights.pth')
        self.gen_optimizer_path = os.path.join(config.PATH, 'optimizer_'+self.name + '.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.transformer_weights_path = os.path.join(config.PATH, self.name + '.pth')
        self.transformer_discriminator_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.reconstructor_weights_path = os.path.join(config.PATH, self.name + '.pth')

    def load(self):
        pass

    def save(self, save_best, psnr, iteration):
        pass




class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM()
        self.hue_loss = HueLoss()

        self.encoder = Encoder(config.BASE_CHANNEL_NUM)
        self.decoder = Decoder(config.BASE_CHANNEL_NUM)


        if config.MODEL == 2:
            #self.feature_processor = DNCNN_FeatureProcessNet(num_features=4*config.BASE_CHANNEL_NUM, num_layers=config.BLOCK_NUM, in_channels=4*config.BASE_CHANNEL_NUM, out_channels=4*config.BASE_CHANNEL_NUM)
            # self.gaussian_filter = GaussianFilter()
            # self.sobel_filter = SobelFilter()
            self.feature_processor = FeatureProcessNet(num_features=4*config.BASE_CHANNEL_NUM, num_blocks=config.BLOCK_NUM )

        self.epoch = 0

        if config.MODEL == 1:
            self.optimizer = optim.Adam(
                [
                    {'params': self.encoder.parameters()},
                    {'params': self.decoder.parameters()}
                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[15000], gamma=0.1,
                                                            last_epoch=self.epoch - 1)
        elif config.MODEL == 2:
            self.optimizer = optim.Adam(
                [
                    {'params': self.feature_processor.parameters()},
                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[15000], gamma=0.1,
                                                            last_epoch=self.epoch - 1)

    def forward_reconstruct(self, x):
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)

        return x

    def forward_restoration(self, x):
        f_corrupted, _ = self.encoder(x)
        f_restored = self.feature_processor(f_corrupted)
        result, _ = self.decoder(f_restored)

        return result, f_restored

    def forward_restoration_sobel_gaussian(self, x):
        f_corrupted, _ = self.encoder(x)
        f_restored, sobel_map, gaussian_map = self.feature_processor(f_corrupted)
        result, _ = self.decoder(f_restored)

        return result, f_restored, sobel_map, gaussian_map


    def process_reconstruct(self, clean_images):
        self.iteration += 1
        self.optimizer.zero_grad()

        outputs = self.forward_reconstruct(clean_images)

        gen_loss = 0
        gen_l2_loss = self.l2_loss(outputs, clean_images)

        gen_loss += 100 * gen_l2_loss

        logs = [
            ("r_l2", gen_l2_loss.item()),
            ("lr", self.get_current_lr()),
        ]

        return outputs, gen_loss, logs

    def process_restoration(self, input_images, clean_images):
        self.iteration += 1
        self.optimizer.zero_grad()

        gt_features, _ = self.encoder(input_images)
        gt_features = gt_features.detach()

        outputs, pred_features = self.forward_restoration(input_images)

        gen_loss = 0

        gen_l2_loss = self.l2_loss(outputs, clean_images)
        gen_fea_loss = self.l1_loss(gt_features, pred_features)
        gen_ssim_loss = 1-self.ssim_loss(outputs,clean_images)
        gen_hue_loss = self.hue_loss(outputs, clean_images)


        gen_loss += 100 * gen_l2_loss
        gen_loss += 0.2 * gen_fea_loss
        gen_loss += 0.5 * gen_ssim_loss
        gen_loss += 0.05 * gen_hue_loss



        logs = [
            ("r_l2", gen_l2_loss.item()),
            ("fea_l1", gen_fea_loss.item()),
            ("l_hue", gen_hue_loss.item()),
            ("l_ssim", gen_ssim_loss.item()),
            ("lr", self.get_current_lr()),
        ]

        return outputs, gen_loss, logs


    def process_restoration_sobel_gaussian(self, input_images, clean_images):
        self.iteration += 1
        self.optimizer.zero_grad()

        gt_gray = self.generate_gray(clean_images)
        gt_sobel = self.sobel_filter.apply_sobel_filter(gt_gray).detach()
        gt_sobel = torch.nn.functional.interpolate(gt_sobel, scale_factor=0.25, mode='bicubic')
        gt_gaussian = self.gaussian_filter.apply_gaussian_filter(gt_gray).detach()
        gt_gaussian = torch.nn.functional.interpolate(gt_gaussian, scale_factor=0.25, mode='bicubic')

        gt_features, _ = self.encoder(input_images)
        gt_features = gt_features.detach()


        outputs, pred_features, pred_sobel, pred_gaussian = self.forward_restoration_sobel_gaussian(input_images)

        gen_loss = 0

        gen_l2_loss = self.l2_loss(outputs, clean_images)
        gen_fea_loss = self.l1_loss(gt_features, pred_features)
        gen_sobel_loss = self.l1_loss(gt_sobel, pred_sobel)
        gen_gaussian_loss = self.l1_loss(gt_gaussian, pred_gaussian)


        gen_loss += 100 * gen_l2_loss
        gen_loss +=  gen_fea_loss
        gen_loss += 0.1 * gen_sobel_loss
        gen_loss += 0.1 * gen_gaussian_loss



        logs = [
            ("r_l2", gen_l2_loss.item()),
            ("fea_l1", gen_fea_loss.item()),
            ("sob_l1",gen_sobel_loss.item()),
            ("gau_l1",gen_gaussian_loss.item()),
            ("lr", self.get_current_lr()),
        ]

        return outputs, gen_loss, logs

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


    # def forward_init(self, noisy_images):
    #     noise = self.dncnn(noisy_images)
    #     return torch.clamp(noisy_images-noise, 0., 1.)
    #
    # def forward_image(self, images, grad):
    #     residual = self.image_net(torch.cat([images, grad], dim=1))
    #     return torch.clamp(images + residual, 0., 1.)
    #
    # def forward_edge(self, images, grad):
    #     residual = self.edge_net(torch.cat([images,grad], dim=1))
    #     #return torch.clamp(refined_edges, 0., 1.)
    #     return torch.clamp(grad + residual, 0., 1.)
    #
    # def process_init(self, noisy_images, clean_images):
    #     self.iteration += 1
    #     self.optimizer.zero_grad()
    #
    #     outputs = self.forward_init(noisy_images)
    #
    #     gen_loss = 0
    #
    #     gen_l2_loss = self.l2_loss(outputs, clean_images)
    #     gen_loss += 100 * gen_l2_loss
    #
    #     # gen_l1_loss = self.l1_loss(outputs,clean_images)
    #     # gen_loss += gen_l1_loss
    #
    #     logs = [
    #         ("r_l2", gen_l2_loss.item()),
    #         ("lr", self.get_current_lr()),
    #     ]
    #
    #     return outputs, gen_loss, logs
    #
    # def process_image(self, input_images, input_grad, clean_images ):
    #     self.iteration += 1
    #     self.optimizer.zero_grad()
    #
    #     outputs = self.forward_image(input_images, input_grad)
    #
    #     gen_loss = 0
    #
    #     gen_l2_loss = self.l2_loss(outputs, clean_images)
    #     gen_loss += 100 * gen_l2_loss
    #
    #     # gen_l1_loss = self.l1_loss(outputs,clean_images)
    #     # gen_loss += gen_l1_loss
    #
    #     logs = [
    #         ("im_l2", gen_l2_loss.item()),
    #         ("lr", self.get_current_lr()),
    #     ]
    #
    #     return outputs, gen_loss, logs
    #
    #
    # def process_edge(self, input_images, input_grad, clean_grad): # loss for edge should be altered?
    #     self.iteration += 1
    #     self.optimizer.zero_grad()
    #
    #     outputs = self.forward_edge(input_images, input_grad)
    #
    #     gen_loss = 0
    #
    #
    #     gen_l1_loss = self.l1_loss(outputs, clean_grad)
    #     gen_loss +=  gen_l1_loss
    #
    #     # gen_l1_loss = self.l1_loss(outputs,clean_images)
    #     # gen_loss += gen_l1_loss
    #
    #     logs = [
    #         ("edge_l1", gen_l1_loss.item()),
    #         ("lr", self.get_current_lr()),
    #     ]
    #
    #     return outputs, gen_loss, logs

    # def forward_transformer(self, input):
    #     in_features, _ = self.encoder(input)
    #     repaired_features, _ = self.feature_repairer(in_features)
    #     outputs, _ = self.decoder(repaired_features)
    #
    #     return outputs, in_features, repaired_features
    #
    #
    #
    # def process_transformer(self, clean_images, noisy_images):
    #     self.iteration += 1
    #     self.optimizer.zero_grad()
    #
    #     outputs, input_noisy_featuers, repaired_noisy_features = self.forward_transformer(noisy_images)
    #
    #     _, gt_input_features = self.forward_reconstructor(clean_images)
    #     gt_input_features = gt_input_features.detach()
    #     gen_loss = 0
    #
    #     gen_feature_restore_loss = self.l1_loss(gt_input_features, repaired_noisy_features)
    #     gen_l2_loss = self.l2_loss(outputs, clean_images)
    #     gen_loss += self.config.TRANSFORMER_RESULT_LOSS_WEIGHT * gen_l2_loss
    #     gen_loss += self.config.TRANSFORMER_REPAIRED_FEATURE_LOSS_WEIGHT * gen_feature_restore_loss
    #
    #     logs = [
    #         ("r_l2", gen_l2_loss.item()),
    #         ("r_fea", gen_feature_restore_loss.item()),
    #         ("lr", self.get_current_lr()),
    #     ]
    #
    #     return outputs, gen_loss, logs

    def save(self, save_best, psnr, iteration):
        if torch.__version__ == '1.6.0':

            if self.config.MODEL == 1:
                torch.save({
                    'encoder':self.encoder.state_dict(),
                    'decoder':self.decoder.state_dict(),
                },self.gen_weights_path[:-4]+'_'+self.name+'.pth' if not save_best else self.gen_weights_path[
                                                                       :-4] +'_'+self.name+ "_%.2f" % psnr + ("_YCbCr" if self.config.PSNR == 'YCbCr' else "_RGB") + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)
            elif self.config.MODEL == 2:
                torch.save({
                           'feature_processor':self.feature_processor.state_dict()
                           },self.gen_weights_path[:-4]+'_'+self.name+'.pth' if not save_best else self.gen_weights_path[
                                                                       :-4] + '_' + self.name + "_%.2f" % psnr + ("_YCbCr" if self.config.PSNR == 'YCbCr' else "_RGB") + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)


            torch.save({
                'iteration': self.iteration,
                'epoch': self.epoch,
                'scheduler': self.scheduler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, self.gen_optimizer_path if not save_best else self.gen_optimizer_path[
                                                           :-4] + "_%.2f" % psnr + ("_YCbCr" if self.config.PSNR == 'YCbCr' else "_RGB") + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)

        else:
            if self.config.MODEL == 1:
                torch.save({
                     'encoder':self.encoder.state_dict(),
                    'decoder':self.decoder.state_dict(),
                },self.gen_weights_path[:-4]+'_'+self.name+'.pth' if not save_best else self.gen_weights_path[
                                                                       :-4] +'_'+self.name+ "_%.2f" % psnr + ("_YCbCr" if self.config.PSNR == 'YCbCr' else "_RGB") + "_%d" % iteration + '.pth')
            elif self.config.MODEL == 2:
                torch.save({
                    'feature_processor': self.feature_processor.state_dict()
                           },self.gen_weights_path[:-4]+'_'+self.name+'.pth' if not save_best else self.gen_weights_path[
                                                                       :-4] + '_' + self.name + "_%.2f" % psnr + ("_YCbCr" if self.config.PSNR == 'YCbCr' else "_RGB") + "_%d" % iteration + '.pth')

            torch.save({
                'iteration': self.iteration,
                'epoch': self.epoch,
                'scheduler': self.scheduler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, self.gen_optimizer_path if not save_best else self.gen_optimizer_path[
                                                           :-4] + "_%.2f" % psnr + ("_YCbCr" if self.config.PSNR == 'YCbCr' else "_RGB") + "_%d" % iteration + '.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path[:-4] + '_reconstruct' + '.pth'):
            print('Loading %s weights...' % 'init')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth',
                                     lambda storage, loc: storage)

            self.encoder.load_state_dict(weights['encoder'])
            self.decoder.load_state_dict(weights['decoder'])

        if self.config.MODEL == 2:
            if os.path.exists(self.gen_weights_path[:-4]+'_'+self.name+'.pth'):
                print('Loading %s weights...' % self.name)

                if torch.cuda.is_available():
                    weights = torch.load(self.gen_weights_path[:-4]+'_'+self.name+'.pth')
                else:
                    weights = torch.load(self.gen_weights_path[:-4]+'_'+self.name+'.pth', lambda storage, loc: storage)

                if self.config.MODEL == 2:
                    self.feature_processor.load_state_dict(weights['feature_processor'])




            # self.residual_finder.load_state_dict(data['residual_finder'])

        if os.path.exists(self.gen_optimizer_path) and self.config.MODE == 1:
            print('Loading %s optimizer...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_optimizer_path)
            else:
                data = torch.load(self.gen_optimizer_path, lambda storage, loc: storage)

            self.optimizer.load_state_dict(data['optimizer'])
            self.scheduler.load_state_dict(data['scheduler'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']

    def backward(self, gen_loss):
        gen_loss.backward()
        self.optimizer.step()

    def update_scheduler(self):
        self.scheduler.step()

    def eval_(self):
        self.encoder.eval()
        self.decoder.eval()

        if self.config.MODEL == 2:
            self.feature_processor.eval()


    def train_(self):
        if self.config.MODEL == 1:
            self.encoder.train()
            self.decoder.train()

        elif self.config.MODEL == 2:
            self.encoder.eval()
            self.decoder.eval()
            self.feature_processor.train()


    def cal_graident(self,x):
        if x.shape[1] == 3:
            x = (0.299 * x[:,0] + 0.587 * x[:,1] + x[:,2] * 0.114).unsqueeze(1)

        g_x = torch.abs(torch.nn.functional.conv2d(x, self.sobelkernel_x, padding=1))
        g_y = torch.abs(torch.nn.functional.conv2d(x, self.sobelkernel_y, padding=1))

        return (g_x + g_y)


    def generate_gray(self, x):
        x = (0.299 * x[:,0] + 0.587 * x[:,1] + x[:,2] * 0.114).unsqueeze(1)
        return x


    def post_process(self,img_high, img_low):
        gray_img = self.generate_gray(img_low)
        result = (1-gray_img)*img_high + (gray_img)*img_low
        return result