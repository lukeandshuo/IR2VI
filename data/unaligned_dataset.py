import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import numpy as np
class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ###TODO dataset dir has been modified
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'night_mwir')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'day_visible')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

        if self.opt.resize_or_crop in ["resize_and_crop_bboxes","object_crop"]:
            self.dir_A_bboxes = os.path.join(opt.dataroot,'annotation',opt.phase,'night_mwir')
            self.dir_B_bboxes = os.path.join(opt.dataroot,'annotation',opt.phase,'day_visible')

            self.A_bboxes_paths = make_dataset(self.dir_A_bboxes)
            self.B_bboxes_paths = make_dataset(self.dir_B_bboxes)

            self.A_bboxes_paths = sorted(self.A_bboxes_paths)
            self.B_bboxes_paths = sorted(self.B_bboxes_paths)

    def bboxes_parser(self,path):
        res = []
        with open(path) as f:
            for line in f:
                line = line.strip().split()
                if line[0] == "%":
                    continue
                else:
                    box = [int(i) for i in line[1:5]]
                    ##bbox format "xywh"
                    ## convert to "xmin,ymin, xmax, ymax"
                    x1, y1, x2, y2 = box
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 =  float(x2)
                    y2 = float(y2)
                    box = np.asarray([x1, y1, x2, y2])
                    res.append(box)
        res = np.asarray(res)
        return res


    def __getitem__(self, index):

        index_A = index % self.A_size
        A_path = self.A_paths[index_A]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.opt.resize_or_crop in ["resize_and_crop_bboxes","object_crop"]:
            A_bboxes_path = self.A_bboxes_paths[index_A]
            B_bboxes_path = self.B_bboxes_paths[index_B]
            A_bboxes = self.bboxes_parser(A_bboxes_path)
            B_bboxes = self.bboxes_parser(B_bboxes_path)
            A = self.transform(A_img,A_bboxes)
            B = self.transform(B_img,B_bboxes)
        else:
            A = self.transform(A_img)
            B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
