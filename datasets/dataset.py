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

    def __init__(self, csv_path, n_aug = 7, n_scale=1, n_distortions=1, patch_size=224, swap_crops=1):

        super().__init__()
        df = pd.read_csv(csv_path)       
        self.image_name  = df['Image_path']
        self.n_aug = n_aug
        self.n_scale = n_scale
        self.n_distortions = n_distortions
        self.patch_size = patch_size
        self.swap = (self.n_aug+1)//2
        self.swap_crops = swap_crops
        self.min_OLA = 0.1
        self.max_OLA = 0.3
        #self.crop_transform()
    def __len__(self):

        return (len(self.image_name))

    def iqa_transformations(self, choice, im, level):

        #level = random.randint(0,4)

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

        elif choice == 23:

            im = imresizedist_bilinear(im,level)

        elif choice == 24:

            im = imresizedist_nearest(im,level)

        elif choice == 25:

            im = imresizedist_lanczos(im,level)

        elif choice == 26:

            im = imblurmotion(im,level)

        else :
            
            pass

        return im

    def choose_y(self):
        low_ = self.patch_size*(1-self.max_OLA)
        high_ = self.patch_size*(1-self.min_OLA)
        y = np.random.randint(low = low_, high = high_)
        # print("choosing Y between: [" , low_, ',', high_, '] -> ', y)
        return y
    
    def choose_x(self, y):
        low_ = np.maximum(0,((1-self.max_OLA)*self.patch_size**2 - self.patch_size*y)/(self.patch_size - y))
        high_ = np.maximum(0,((1-self.min_OLA)*self.patch_size**2 - self.patch_size*y)/(self.patch_size - y))
        x = np.random.randint(low = low_, high = high_)
        # print("choosing X between: [" , low_, ',', high_, '] -> ', x)
        return x

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
        #print(self.image_name[idx])

        # if self.n_scale == 2:
        #     image_half = image.resize((image.size[0]//2,image.size[1]//2))
        
        ## create  positive pair
        img_pair1 = transforms.ToTensor()(image)  # 1, 3, H, W
        chunk1 = img_pair1.unsqueeze(0)
        #img_pair2 = transforms.ToTensor()(image)  # 1, 3, H, W
        #chunk2 = img_pair2.unsqueeze(0)

        choices = list(range(1, 27))
        random.shuffle(choices)
        for i in range(0,self.n_aug):
            if self.n_distortions == 1:
                ## generate self.aug distortion-augmentations
                level = random.randint(0,4)
                img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[i], image, level))
                img_aug_i = img_aug_i.unsqueeze(0)
                chunk1 = torch.cat([chunk1, img_aug_i], dim=0)
                #chunk2 = torch.cat([chunk2, img_aug_i], dim=0)
            else :
                j = random.randint(0,25)
                if random.random()>0.1:
                    level = random.randint(0,4)
                    img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[i], image, level))
                else:
                    level1 = random.randint(0,4)
                    level2 = random.randint(0,level1)
                    level1 = level1-level2
                    img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[j], self.iqa_transformations(choices[i], image, level1),level2))
                img_aug_i = img_aug_i.unsqueeze(0)
                chunk1 = torch.cat([chunk1, img_aug_i], dim=0)
                #chunk2 = torch.cat([chunk2, img_aug_i], dim=0)

        # chunk1, chunk2  -> self.n_aug+1 , 3, H, W

        # generate two random crops
        #chunk1_1 = self.crop_transform(chunk1,self.patch_size)
        #chunk1_2 = self.crop_transform(chunk1,self.patch_size)

        #chunk1, chunk2  -> self.n_aug+1 , 3, 256 , 256

        #temp = chunk1_1[0]
        #chunk1_1[0] = chunk1_2[0]
        #chunk1_2[0] = temp

        start_y = self.choose_y()
        start_x = self.choose_x(start_y)
        chunk1_1 = self.crop_transform(chunk1, crop_size = self.patch_size+max(start_y, start_x))

        chunk1_2 = chunk1_1[:,:,start_y:start_y+self.patch_size, start_x:start_x+self.patch_size]
        chunk1_1 = chunk1_1[:,:,:self.patch_size,:self.patch_size]



        if self.swap_crops == 1:
            chunk1_1[0:self.swap] = chunk1_1[0:self.swap] + chunk1_2[0:self.swap]            ## x = x + y
            chunk1_2[0:self.swap] = chunk1_1[0:self.swap] - chunk1_2[0:self.swap]            ## y = x - y
            chunk1_1[0:self.swap] = chunk1_1[0:self.swap] - chunk1_2[0:self.swap]            ## x = x - y
        t1 =  torch.cat((chunk1_1, chunk1_2), dim=1)

        if self.n_scale == 2:

            ## create  positive pair
            # img_pair1 = transforms.ToTensor()(image_half)  # 1, 3, H, W
            # chunk3 = img_pair1.unsqueeze(0)
            # img_pair2 = transforms.ToTensor()(image_half)  # 1, 3, H, W
            # chunk4 = img_pair2.unsqueeze(0)

            

            # choices = list(range(1, 23))
            # random.shuffle(choices)
            # for i in range(0,self.n_aug):
            #     ## generate self.aug distortion-augmentations
            #     img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[i], image_half))
            #     img_aug_i = img_aug_i.unsqueeze(0)
            #     chunk3 = torch.cat([chunk3, img_aug_i], dim=0)
            #     chunk4 = torch.cat([chunk4, img_aug_i], dim=0)

            # # chunk3, chunk4  -> self.n_aug+1 , 3, H/2, W/2

            chunk3 = torch.nn.functional.interpolate(chunk1,size=(chunk1.shape[2]//2,chunk1.shape[3]//2),mode='bicubic',align_corners=True)
            #chunk4 = torch.nn.functional.interpolate(chunk2,size=(chunk2.shape[2]//2,chunk2.shape[3]//2),mode='bicubic',align_corners=True)
            # generate two random crops
            #chunk2_1 = self.crop_transform(chunk3,self.patch_size)
            #chunk2_2 = self.crop_transform(chunk3,self.patch_size)

            start_y = self.choose_y()
            start_x = self.choose_x(start_y)
            chunk2_1 = self.crop_transform(chunk3, crop_size = self.patch_size+max(start_y, start_x))

            chunk2_2 = chunk2_1[:,:,start_y:start_y+self.patch_size, start_x:start_x+self.patch_size]
            chunk2_1 = chunk2_1[:,:,:self.patch_size,:self.patch_size]

            #chunk1, chunk2  -> self.n_aug+1 , 3, 256 , 256

            #temp = chunk2_1[0]
            #chunk2_1[0] = chunk2_2[0]
            #chunk2_2[0] = temp

            if self.swap_crops == 1:
                chunk2_1[self.swap:] = chunk2_1[self.swap:] + chunk2_2[self.swap:]            ## x = x + y
                chunk2_2[self.swap:] = chunk2_1[self.swap:] - chunk2_2[self.swap:]            ## y = x - y
                chunk2_1[self.swap:] = chunk2_1[self.swap:] - chunk2_2[self.swap:] 
            t2 = torch.cat((chunk2_1, chunk2_2), dim=1)

        if self.n_scale == 1:
            return t1
        else:
            return torch.cat((t1, t2), dim=0)