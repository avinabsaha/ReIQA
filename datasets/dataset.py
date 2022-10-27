from __future__ import print_function

import numpy as np
import torch
from torchvision import datasets
from torch.utils import data
import pandas as pd
from .iqa_distortions import *
import random
from torchvision import transforms



class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, index


class IQAImageClass(data.Dataset):

    def __init__(self, csv_path, n_aug = 7, n_scale=1):

        super().__init__()
        df = pd.read_csv(csv_path)       
        self.image_name  = df['Image_path']
        self.n_aug = n_aug
        self.n_scale = n_scale
        #self.crop_transform()
    def __len__(self):

        return (len(self.image_name))

    def iqa_transformations(self, choice, im):

        level = random.randint(0,4)

        if choice == 1:

            im = imblurgauss(im, level)
 
        elif choice == 2 :
        
            im = imblurlens(im,level)
        
        elif choice == 3 :

            im = imcolordiffuse(im,level)

        elif choice == 4 :

            im = imcolorshift(im,level)

        elif choice == 5 :

            im = imcolorsaturate(im,level)

        elif choice == 6 :

            im = imsaturate(im,level)

        elif choice == 7 :

            im = imcompressjpeg(im,level)

        elif choice == 8 :

            im = imnoisegauss(im,level)

        elif choice == 9 :

            im = imnoisecolormap(im,level)

        elif choice == 10 :

            im = imnoiseimpulse(im,level)

        elif choice == 11 :

            im = imnoisemultiplicative(im,level)

        elif choice == 12 :

            im = imdenoise(im,level)

        elif choice == 13 :

            im = imbrighten(im,level)

        elif choice == 14 :

            im = imdarken(im, level)

        elif choice == 15 :

            im = immeanshift(im,level)

        elif choice == 16 :

            im = imresizedist(im,level)

        elif choice == 17 :

            im = imsharpenHi(im,level)

        elif choice == 18 :

            im = imcontrastc(im,level)

        elif choice == 19 :

            im = imcolorblock(im,level)

        elif choice == 20 :

            im = impixelate(im,level)

        elif choice == 21 :

            im = imnoneccentricity(im,level)

        elif choice == 22 :

            im = imjitter(im,level)

        else :
            
            pass

        return im

    def crop_transform(self, image, crop_size=224):

        #print(image.shape)
        if image.shape[2] < crop_size or image.shape[3] < crop_size :
        #if crop_type == 'center':
            image = transforms.transforms.CenterCrop(crop_size)(image)
        else : # crop_type == 'random':
            image = transforms.transforms.RandomCrop(crop_size)(image)




        return image

    def __getitem__(self,idx) :

        image = Image.open(self.image_name[idx]).convert('RGB')

        if self.n_scale == 2:
            image_half = image.resize((image.size[0]//2,image.size[1]//2))
        


        ## create  positive pair
        img_pair1 = transforms.ToTensor()(image)  # 1, 3, H, W
        chunk1 = img_pair1.unsqueeze(0)
        img_pair2 = transforms.ToTensor()(image)  # 1, 3, H, W
        chunk2 = img_pair2.unsqueeze(0)

        

        choices = list(range(1, 23))
        random.shuffle(choices)
        for i in range(0,self.n_aug):
            ## generate self.aug distortion-augmentations
            img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[i], image))
            img_aug_i = img_aug_i.unsqueeze(0)
            chunk1 = torch.cat([chunk1, img_aug_i], dim=0)
            chunk2 = torch.cat([chunk2, img_aug_i], dim=0)

        # chunk1, chunk2  -> self.n_aug+1 , 3, H, W

        # generate two random crops
        chunk1 = self.crop_transform(chunk1)
        chunk2 = self.crop_transform(chunk2)

        #chunk1, chunk2  -> self.n_aug+1 , 3, 256 , 256

        temp = chunk1[0]
        chunk1[0] = chunk2[0]
        chunk2[0] = temp
        t1 =  torch.cat((chunk1, chunk2), dim=1)

        if self.n_scale == 2:

            ## create  positive pair
            img_pair1 = transforms.ToTensor()(image_half)  # 1, 3, H, W
            chunk3 = img_pair1.unsqueeze(0)
            img_pair2 = transforms.ToTensor()(image_half)  # 1, 3, H, W
            chunk4 = img_pair2.unsqueeze(0)

            

            choices = list(range(1, 23))
            random.shuffle(choices)
            for i in range(0,self.n_aug):
                ## generate self.aug distortion-augmentations
                img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[i], image_half))
                img_aug_i = img_aug_i.unsqueeze(0)
                chunk3 = torch.cat([chunk3, img_aug_i], dim=0)
                chunk4 = torch.cat([chunk4, img_aug_i], dim=0)

            # chunk3, chunk4  -> self.n_aug+1 , 3, H/2, W/2

            # generate two random crops
            chunk3 = self.crop_transform(chunk3)
            chunk4 = self.crop_transform(chunk4)

            #chunk1, chunk2  -> self.n_aug+1 , 3, 256 , 256

            temp = chunk3[0]
            chunk3[0] = chunk4[0]
            chunk4[0] = temp
            t2 = torch.cat((chunk3, chunk4), dim=1)

            if self.n_scale == 1:
                return t1
            else:
                return torch.cat((t1, t2), dim=1)
