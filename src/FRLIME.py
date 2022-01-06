import os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import  Model
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import  EdgeAccuracy, PSNR_RGB, PSNR_YCbcr



class FRLIME():
    def __init__(self, config):
        self.config = config

        self.model = Model(config).to(config.DEVICE)


        if config.PSNR == 'RGB':
            self.psnr = PSNR_RGB(255.0).to(config.DEVICE)
        elif config.PSNR == 'YCbCr':
            self.psnr = PSNR_YCbcr().to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config,crop_size=None, clean_flist=config.TEST_CLEAN_FLIST, noisy_flist= config.TEST_NOISY_FLIST, augment=False, split='test', preload_to_memory=False)
        else:
            self.train_dataset = Dataset(config, crop_size=config.CROP_SIZE, clean_flist=config.TRAIN_CLEAN_FLIST, noisy_flist=config.TRAIN_NOISY_FLIST,  augment=True, preload_to_memory=False, split='train')
            #self.val_dataset = Dataset(config, crop_size=256, clean_flist=config.VAL_FLIST,  augment=False, preload_to_memory=False, split='validate')
            self.val_dataset = Dataset(config, crop_size=256, clean_flist=config.TEST_CLEAN_FLIST,noisy_flist=config.TEST_NOISY_FLIST,  augment=False, preload_to_memory=False, split='validate')

            self.test_dataset = Dataset(config, crop_size=None, clean_flist=config.TEST_CLEAN_FLIST,
                                        noisy_flist=config.TEST_NOISY_FLIST, augment=False, split='test', preload_to_memory=False)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model.name + '.dat')

    def load(self):
        self.model.load()


    def save(self, save_best=False, psnr=None, iteration=None):
        self.model.save(save_best,psnr,iteration)



    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=16,
            drop_last=True,
            shuffle=True
        )


        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        epoch = self.model.epoch
        highest_psrn = 0
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            print('epoch:', epoch)

            for items in train_loader:
                self.model.train_()
                #self.model.eval_()
                clean_images, noisy_images = self.cuda(*items)


                if model == 1:
                    outputs, loss, logs = self.model.process_reconstruct(clean_images)

                    psnr = self.psnr(self.postprocess(clean_images), self.postprocess(outputs))
                    mae = torch.mean((torch.abs(clean_images - outputs)))
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.model.backward(loss)

                    outputs, loss, _ = self.model.process_reconstruct(noisy_images)

                    self.model.backward(loss)

                    psnr = self.psnr(self.postprocess(noisy_images), self.postprocess(outputs))
                    mae = torch.mean((torch.abs(noisy_images - outputs)))
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    iteration = self.model.iteration

                elif model == 2:


                    outputs, loss, logs = self.model.process_restoration(noisy_images, clean_images)


                    psnr = self.psnr(self.postprocess(clean_images), self.postprocess(outputs))
                    mae = torch.mean((torch.abs(clean_images - outputs)))
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.model.backward(loss)

                    iteration = self.model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(clean_images), values=logs if self.config.VERBOSE else [x for x in logs ])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    current_psnr = self.eval()
                    if current_psnr > highest_psrn:
                        print('\nnew high accuracy:', current_psnr)
                        highest_psrn = current_psnr
                        self.save(save_best=True, psnr=current_psnr, iteration=iteration)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

            # update epoch for scheduler
            self.model.epoch = epoch
            self.model.update_scheduler()

        print('\nEnd training....')

    def eval(self): ############ need to change!
        val_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            drop_last=False,
            shuffle=False
        )

        model = self.config.MODEL
        total = len(self.test_dataset)

        self.model.eval_()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        psnrs=[]
        with torch.no_grad():
            for items in val_loader:
                iteration += 1
                clean_images, noisy_images = self.cuda(*items)

                if model == 1:
                    h, w = noisy_images.shape[2:4]
                    predicted_results = self.model.forward_reconstruct(clean_images)
                    predicted_results = self.crop_result(predicted_results, h, w)
                    psnr = self.psnr(self.postprocess(clean_images), self.postprocess(predicted_results))

                    psnrs.append(psnr.item())
                    logs = []
                    logs.append(('psnr_ycbcr' if self.config.PSNR == 'YCbCr' else 'psnr_rgb', psnr.item()))


                if model == 2:

                    h, w = noisy_images.shape[2:4]
                    noisy_input_images = noisy_images
                    predicted_results, _ = self.model.forward_restoration(noisy_input_images)
                    predicted_results = self.crop_result(predicted_results, h, w)
                    psnr = self.psnr(self.postprocess(clean_images), self.postprocess(predicted_results))

                    psnrs.append(psnr.item())
                    logs = []
                    logs.append(('psnr_ycbcr' if self.config.PSNR == 'YCbCr' else 'psnr_rgb', psnr.item()))


                logs = [("it", iteration), ] + logs
                progbar.add(len(noisy_images), values=logs)

        return np.mean(psnrs)

    def test(self):
        model = self.config.MODEL
        self.model.eval_()
        create_dir(self.results_path)
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0

        psnrs = []
        with torch.no_grad():

            for items in test_loader:
                name = self.test_dataset.load_name(index)[:-4]+'.png'

                index += 1



                if model == 1:
                    clean_images, noisy_images = self.cuda(*items)
                    ## check if the input size is multiple of 4
                    h, w = noisy_images.shape[2:4]
                    #start_time = time.time() 
                    predicted_results = self.model.forward_reconstruct(clean_images)
                    #end_time = time.time()
                    #runtime = str((end_time-start_time).seconds)
                    #print('runtime: %.2f sec'%(end_time-start_time) )

                    images_gather = stitch_images(
                        self.postprocess(clean_images),
                        self.postprocess(predicted_results),
                        img_per_row=1
                    )



                    path = os.path.join(self.results_path, self.model.name + '_gather')
                    create_dir(path)
                    save_name = os.path.join(path, name)
                    images_gather.save(save_name)


                    psnr = self.psnr(self.postprocess(predicted_results), self.postprocess(clean_images))
                    psnrs.append(psnr.item())
                    print('PSNR_YCbCr:' if self.config.PSNR == 'YCbCr' else 'PSNR_RGB:', psnr)




                elif model == 2:
                    if self.config.HAS_GT == 1:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)

                        clean_images, noisy_images = self.cuda(*items)
                        h, w = noisy_images.shape[2:4]
                        noisy_input_images = self.pad_input(noisy_images)
                        #gt_features, _ = self.model.encoder(clean_images)
                        #start = time.time()
                        start.record()
                        predicted_results, pred_features = self.model.forward_restoration(noisy_input_images)
                        #end.record()
                        #torch.cuda.synchronize()
                        #print(start.elapsed_time(end))
                       # print('time: %.4f sec'%(end-start))
                        predicted_results = self.crop_result(predicted_results, h, w)
                        post_processed_results = self.model.post_process(img_high=predicted_results, img_low=noisy_images)
                        end.record()
                        torch.cuda.synchronize()
                        print('time:'+str(start.elapsed_time(end))+' ms')

                        # average_gt_features = torch.mean(gt_features, dim=1, keepdim=True)
                        # average_pred_features = torch.mean(pred_features, dim=1, keepdim=True)

                        # images_gather = stitch_images(
                        #     self.postprocess(clean_images),
                        #     self.postprocess(noisy_images),
                        #     #self.generate_color_map(average_gt_features,[h,w]),
                        #     #self.generate_color_map(average_pred_features, [h,w]),
                        #     self.postprocess(predicted_results),
                        #     self.postprocess(post_processed_results),
                        #     img_per_row=1
                        # )





                        psnr = self.psnr(self.postprocess(predicted_results), self.postprocess(clean_images))
                        psnrs.append(psnr.item())
                        print('PSNR_YCbCr:' if self.config.PSNR == 'YCbCr' else 'PSNR_RGB:', psnr)

                        path = os.path.join(self.results_path, self.model.name + '_features')
                        create_dir(path)


                       # print(torch.mean(predicted_results))
                        #print(torch.var(predicted_results))
                    elif self.config.HAS_GT == 0:
                        noisy_images = items.to(self.config.DEVICE)

                        h, w = noisy_images.shape[2:4]
                        noisy_input_images = self.pad_input(noisy_images)
                        #gt_features, _ = self.model.encoder(clean_images)
                        predicted_results, pred_features = self.model.forward_restoration(noisy_input_images)

                        predicted_results = self.crop_result(predicted_results, h, w)
                        post_processed_results = self.model.post_process(img_high=predicted_results,
                                                                         img_low=noisy_images)


                    path = os.path.join(self.results_path, self.model.name + '_gather')
                    create_dir(path)
                    # save_name = os.path.join(path,name)
                    # images_gather.save(save_name)




                if self.config.HAS_GT == 1:
                    output = self.postprocess(predicted_results)[0]
                    path = os.path.join(self.results_path, self.model.name, name)
                    if not os.path.exists(os.path.join(self.results_path,self.model.name)):
                        os.mkdir(os.path.join(self.results_path,self.model.name))
                    print(index, name)

                    imsave(output, path)
                elif self.config.HAS_GT == 0:
                    output = self.postprocess(post_processed_results)[0]
                    path = os.path.join(self.results_path, self.model.name, name)
                    if not os.path.exists(os.path.join(self.results_path, self.model.name)):
                        os.mkdir(os.path.join(self.results_path, self.model.name))

                    print(index, name)
                    imsave(output, path)




            print('Total PSNR_'+('YCbCr:' if self.config.PSNR =='YCbCr' else 'RGB:'), np.mean(psnrs))
            print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return
        self.model.eval_()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        clean_images, noisy_images = self.cuda(*items)




        # inpaint with edge model / joint model
        with torch.no_grad():
            iteration = self.model.iteration

            if model == 1:

                h, w = noisy_images.shape[2:4]

                ## check if the input size is multiple of 4
                predicted_results = self.model.forward_reconstruct(clean_images)

                images_sample = stitch_images(
                    self.postprocess(clean_images),
                    self.postprocess(predicted_results),
                    img_per_row=1
                )


            elif model == 2:


                h, w = noisy_images.shape[2:4]
                noisy_input_images = noisy_images
                gt_features, _ = self.model.encoder(clean_images)
                predicted_results, pred_features = self.model.forward_restoration(noisy_input_images)
                predicted_results = self.crop_result(predicted_results, h, w)
                average_gt_features = torch.mean(gt_features, dim=1, keepdim=True)
                average_pred_features = torch.mean(pred_features, dim=1 , keepdim=True)

                noisy_features,_ = self.model.encoder(noisy_input_images)
                noisy_results,_ = self.model.decoder(noisy_features)


                images_sample = stitch_images(
                    self.postprocess(clean_images),
                    self.postprocess(noisy_images),
                    self.postprocess(predicted_results),
                    self.postprocess(noisy_results),
                    self.postprocess(torch.abs(noisy_images-predicted_results)),
                    self.postprocess(average_gt_features, size=[256,256]),
                    self.postprocess(average_pred_features, size=[256,256]),

                    img_per_row=1
                )




            path = os.path.join(self.samples_path, self.model.name)
            name = os.path.join(path, str(iteration).zfill(5) + ".png")
            create_dir(path)
            print('\nsaving sample ' + name)
            images_sample.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img, size=None):
        # [0, 1] => [0, 255]
        if size is not None:
            img = torch.nn.functional.interpolate(img,size,mode='bicubic')
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def generate_color_map(self, imgs, size=[256,256]):
        # N 1 H W -> N H W 3 color map
        imgs = (imgs*255.0).int().squeeze(1).cpu().numpy().astype(np.uint8)
        N, height,width = imgs.shape

        colormaps = np.full((N,size[0],size[1],3),1)

        for i in range(imgs.shape[0]):
            colormaps[i] = cv2.resize((cv2.applyColorMap(imgs[i], cv2.COLORMAP_JET)),(size[1],size[0]))

        #transfer to tensor than to gpu
        #firstly the channel BGR->RGB
        colormaps = colormaps[...,[2,1,0]]

        #than to tensor, to gpu
        colormaps = torch.from_numpy(colormaps).cuda()

        return colormaps

    def crop_result(self, result, input_h, input_w):
        crop_h = crop_w = 0

        if input_h % 4 != 0:
            crop_h = 4 - (input_h % 4)

        if input_w % 4 != 0:
            crop_w = 4 - (input_w % 4)

        if crop_h != 0:
            result = result[...,:-crop_h, :]
        if crop_w != 0:
            result = result[...,:-crop_w]
        return result

    def pad_input(self, input):
        input_h, input_w = input.shape[2:]
        pad_h = pad_w = 0

        if input_h % 4 != 0:
            pad_h = 4 - (input_h % 4)

        if input_w % 4 != 0:
            pad_w = 4 - (input_w % 4)

        #print(pad_h, pad_w)

        input = torch.nn.functional.pad(input, (0,pad_w, 0, pad_h), mode='constant', value=0)

        return input
