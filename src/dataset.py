import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, crop_size,  noisy_flist, clean_flist=None, augment=True, preload_to_memory=True, split='train'):
        super(Dataset, self).__init__()
        self.augment = augment
        self.config = config
        assert split in ['train', 'test', 'validate']

        self.split = split

        if preload_to_memory and self.split in ['train', 'validate']:

            self.clean_data = self.load_image_to_memory(clean_flist)
            self.noisy_data = self.load_image_to_memory(noisy_flist)

        else:
            if self.config.HAS_GT == 1:
                self.clean_data = self.load_flist(clean_flist)
            self.noisy_data = self.load_flist(noisy_flist)


        self.input_size = crop_size
        self.sigma = config.SIGMA # noise level

        self.preload_to_memory = preload_to_memory

        # self.transforms = transforms.Compose(([
        #     transforms.RandomCrop((self.input_size,self.input_size)),
        # ] if self.split in ['train', 'validate'] else [] )
        #  + ([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        #      transforms.Lambda(lambda img: self.RandomRot(img))]
        #      if self.augment else [])
        #   + [transforms.ToTensor()]
        # )





        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        #if config.MODE == 2:
        #    self.mask = 6

    def __len__(self):
        return len(self.clean_data) if self.config.HAS_GT == 1 else len(self.noisy_data)

    def __getitem__(self, index):
        # try:
        item = self.load_item(index)
        # except:
        #     print('loading error: ' + self.data[index])
        #     item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.clean_data[index] if self.config.HAS_GT == 1 else self.noisy_data[index]
        return os.path.basename(name)

    def load_item(self, index):

        # load image
        if self.split in ['train', 'validate','test']:
            if self.config.HAS_GT == 1:
                if self.preload_to_memory:
                    img_clean = Image.fromarray(self.clean_data[index])
                    img_noisy = Image.fromarray(self.noisy_data[index])
                else:
                    img_clean = Image.open(self.clean_data[index])
                    img_noisy = Image.open(self.noisy_data[index])

                img_clean, img_noisy = self.apply_transforms(img_clean, img_noisy)

                return img_clean, img_noisy
            else:
                if self.preload_to_memory:
                    #img_clean = Image.fromarray(self.clean_data[index])
                    img_noisy = Image.fromarray(self.noisy_data[index])
                else:
                    #img_clean = Image.open(self.clean_data[index])

                    img_noisy = Image.open(self.noisy_data[index])

                img_noisy = self.apply_transforms(img_noisy)[0]
                return img_noisy

        # else:
        #     img_clean = Image.open(self.clean_data[index])
        #     img_noisy = Image.open(self.noisy_data[index])
        #
        #
        #
        #     img_clean = self.transforms(img_clean)
        #     img_noisy = self.transforms(img_noisy)

            # noise = torch.normal(torch.zeros(img_clean.size()), self.sigma / 255.0)
            #
            # img_noisy = img_clean + noise
            # img_noisy = torch.clamp(img_noisy, 0., 1.)

        #return img_clean, img_noisy




    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                #try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                #except:
                    #return [flist]

        return []


    def load_image_to_memory(self, flist):
        filelist = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        images_list = []
        for i in range(len(filelist)):
            images_list.append(np.array(Image.open(filelist[i])))
            if i%100 == 0:
                print('loading data: %d / %d', i+1, len(filelist))
        return images_list

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def RandomRot(self, img, angle=90, p=0.5):
        if random.random() > p:
            return transforms.functional.rotate(img, angle)
        return img


    def apply_transforms(self, *imgs):
        # self.transforms = transforms.Compose(([
        #                                           transforms.RandomCrop((self.input_size, self.input_size)),
        #                                       ] if self.split in ['train', 'validate'] else [])
        #                                      + ([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        #                                          transforms.Lambda(lambda img: self.RandomRot(img))]
        #                                         if self.augment else [])
        #                                      + [transforms.ToTensor()]
        #                                      )

        imgs = list(imgs)

        if self.split in ['train','validate']:# and self.config.HAS_GT == 1:
            # print(clean_img.size)
            # print(self.input_size)

            # if imgs[0].size != imgs[1].size:
            #     print('fuck')

            i,j,h,w = transforms.RandomCrop.get_params(imgs[0], (self.input_size, self.input_size))

            for it in range(len(imgs)):
                imgs[it] = F.crop(imgs[it], i,j,h,w)
            # clean_img = TF.crop(clean_img, i,j,h,w)
            # noisy_img = TF.crop(noisy_img, i,j,h,w)

            if self.augment:
                if random.random()>0.5:
                    for i in range(len(imgs)):
                        imgs[i] = F.hflip(imgs[i])

                if random.random() > 0.5:
                    for i in range(len(imgs)):
                        imgs[i] = F.vflip(imgs[i])




        if self.split in ['test']:
            if len(imgs) > 1:
                if imgs[0].size != imgs[1].size:
                    if imgs[0].size[0]>imgs[1].size[0]:
                        imgs[0] = F.center_crop(imgs[0], list(imgs[1].size)[::-1])
                    else:
                        imgs[1] = F.center_crop(imgs[1], list(imgs[0].size)[::-1])



        for i in range(len(imgs)):
            imgs[i] = F.to_tensor(imgs[i])

        return imgs