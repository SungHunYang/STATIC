import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random
import cv2

from utils import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   # num_workers=args.num_threads,
                                   num_workers=6,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':
            if self.args.dataset == 'kitti':
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]
                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file.replace('image_02', 'image_03')
                    depth_file.replace('image_02', 'image_03')
            else:
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]

            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)
            fill_path = depth_path.replace('groundtruth', 'fill')

            image_idx = int(rgb_file.split('/')[-1].split('.')[0])
            next_image_idx = image_idx + 1
            next_rgb_file = rgb_file.replace(str(image_idx).zfill(10), str(next_image_idx).zfill(10))
            next_depth_file = depth_file.replace(str(image_idx).zfill(10), str(next_image_idx).zfill(10))
            next_image_path = os.path.join(self.args.data_path, next_rgb_file)
            next_depth_path = os.path.join(self.args.gt_path, next_depth_file)
            next_fill_path = next_depth_path.replace('groundtruth', 'fill')

            if not os.path.exists(next_image_path) or not os.path.exists(next_depth_path) or not os.path.exists(next_fill_path):
                return self.__getitem__(random.randint(0, len(self.filenames) - 1))
    
            image = Image.open(image_path)
            next_image = Image.open(next_image_path)
            depth_gt = Image.open(depth_path)
            next_depth_gt = Image.open(next_depth_path)
            fill_gt = Image.open(fill_path)
            next_fill_gt = Image.open(next_fill_path)
            
            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                next_depth_gt = next_depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                next_image = next_image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                fill_gt = fill_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                next_fill_gt = next_fill_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            
            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    valid_mask = np.zeros_like(depth_gt)
                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask==0] = 0
                    depth_gt = Image.fromarray(depth_gt)
                else:
                    depth_gt = depth_gt.crop((43, 45, 608, 472))
                    image = image.crop((43, 45, 608, 472))
    
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                next_image = self.rotate_image(next_image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                next_depth_gt = self.rotate_image(next_depth_gt, random_angle, flag=Image.NEAREST)
                fill_gt = self.rotate_image(fill_gt, random_angle, flag=Image.NEAREST)
                next_fill_gt = self.rotate_image(next_fill_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            next_image = np.asarray(next_image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            next_depth_gt = np.asarray(next_depth_gt, dtype=np.float32)
            next_depth_gt = np.expand_dims(next_depth_gt, axis=2)
            fill_gt = np.asarray(fill_gt, dtype=np.float32)
            fill_gt = np.expand_dims(fill_gt, axis=2)
            next_fill_gt = np.asarray(next_fill_gt, dtype=np.float32)
            next_fill_gt = np.expand_dims(next_fill_gt, axis=2)

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
                img, depth = image, depth_gt
                #<https://arxiv.org/abs/2107.07684>
                H, W = img.shape[0], img.shape[1]
                a, b, c, d = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
                l, u = int(a*W), int(b*H)
                w, h = int(max((W-a*W)*c*0.75, 1)), int(max((H-b*H)*d*0.75, 1))
                depth_copied = np.repeat(depth, 3, axis=2)
                M = np.ones(img.shape)
                M[l:l+h, u:u+w, :] = 0
                img = M*img + (1-M)*depth_copied
                image = img.astype(np.float32)
            else:
                depth_gt = depth_gt / 256.0
                next_depth_gt = next_depth_gt / 256.0

            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                image, next_image, depth_gt, next_depth_gt, fill_gt, next_fill_gt = self.random_crop(image, next_image, depth_gt, next_depth_gt, fill_gt, next_fill_gt, self.args.input_height, self.args.input_width)
            image, next_image, depth_gt, next_depth_gt, fill_gt, next_fill_gt = self.train_preprocess(image, next_image, depth_gt, next_depth_gt, fill_gt, next_fill_gt)
            sample = {'image': image, 'next_image': next_image, 'depth': depth_gt, 'next_depth': next_depth_gt, 'fill': fill_gt, 'next_fill': next_fill_gt, 'focal': focal}
        
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image_idx = int(image_path.split('/')[-1].split('.')[0])
            next_image_idx = image_idx + 1
            next_image_path = image_path.replace(str(image_idx).zfill(10), str(next_image_idx).zfill(10))

            if not os.path.exists(next_image_path):
                next_image_path = image_path

            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                if self.args.dataset == 'kitti':
                    depth_path = os.path.join(gt_path, sample_path.split()[1])
                    next_depth_path = depth_path.replace(str(image_idx).zfill(10), str(next_image_idx).zfill(10))
                    fill_path = depth_path.replace('groundtruth', 'fill')
                    next_fill_path = next_depth_path.replace('groundtruth', 'fill')
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    fill_gt = Image.open(fill_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    fill_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if not os.path.exists(next_image_path) or not os.path.exists(next_depth_path) or not os.path.exists(next_fill_path):
                    next_image_path = image_path
                    next_depth_path = depth_path
                    next_fill_path = fill_path
                next_image = np.asarray(Image.open(next_image_path), dtype=np.float32) / 255.0
                if has_valid_depth:
                    next_depth_gt = Image.open(next_depth_path)
                    next_fill_gt = Image.open(next_fill_path)
                else:
                    next_depth_gt = False
                    next_fill_gt = False

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    next_depth_gt = np.asarray(next_depth_gt, dtype=np.float32)
                    next_depth_gt = np.expand_dims(next_depth_gt, axis=2)
                    fill_gt = np.asarray(fill_gt, dtype=np.float32)
                    fill_gt = np.expand_dims(fill_gt, axis=2)
                    next_fill_gt = np.asarray(next_fill_gt, dtype=np.float32)
                    next_fill_gt = np.expand_dims(next_fill_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0
                        next_depth_gt = next_depth_gt / 256.0

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                next_image = next_image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    next_depth_gt = next_depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    fill_gt = fill_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    next_fill_gt = next_fill_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            if self.mode == 'online_eval':
                sample = {'image': image, 'next_image': next_image, 'depth': depth_gt, 'next_depth': next_depth_gt, 'fill': fill_gt, 'next_fill': next_fill_gt, 'focal': focal, 'has_valid_depth': has_valid_depth, 'path': image_path}
            else:
                sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, next_img, depth, next_depth, fill, next_fill, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        next_img = next_img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        next_depth = next_depth[y:y + height, x:x + width, :]
        fill = fill[y:y + height, x:x + width, :]
        next_fill = next_fill[y:y + height, x:x + width, :]
        return img, next_img, depth, next_depth, fill, next_fill

    def train_preprocess(self, image, next_image, depth_gt, next_depth_gt, fill_gt, next_fill_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            next_image = (next_image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            next_depth_gt = (next_depth_gt[:, ::-1, :]).copy()
            fill_gt = (fill_gt[:, ::-1, :]).copy()
            next_fill_gt = (next_fill_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
            next_image = self.augment_image(next_image)
    
        return image, next_image, depth_gt, next_depth_gt, fill_gt, next_fill_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, next_image = sample['image'], sample['next_image']
        focal = sample['focal']

        image = self.to_tensor(image)
        image = self.normalize(image)

        next_image = self.to_tensor(next_image)
        next_image = self.normalize(next_image)

        if self.mode == 'test':
            return {'image': image, 'next_image': next_image, 'focal': focal}

        depth = sample['depth']
        next_depth = sample['next_depth']
        fill = sample['fill']
        next_fill = sample['next_fill']

        if self.mode == 'train':
            depth = self.to_tensor(depth)
            next_depth = self.to_tensor(next_depth)
            fill = self.to_tensor(fill)
            next_fill = self.to_tensor(next_fill)
            return {'image': image, 'next_image': next_image, 'depth': depth, 'next_depth': next_depth, 'fill': fill, 'next_fill': next_fill, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'next_image': next_image, 'depth': depth, 'next_depth': next_depth, 'fill': fill, 'next_fill': next_fill, 'focal': focal, 'has_valid_depth': has_valid_depth, 'path': sample['path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
