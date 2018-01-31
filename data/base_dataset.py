import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from numpy import random
import cv2
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'resize_and_crop_bboxes':
        transform_list.append(Resize((opt.loadSize,opt.loadSize)))
        transform_list += [RandomSampleCrop_FixedSize((opt.fineSize,opt.fineSize)),
                           ToTensor(),
                           Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
        return Compose(transform_list)
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'object_crop':
        transform_list.append(ObjectCrop())
        transform_list += [Resize((opt.fineSize,opt.fineSize)),
                           ToTensor(),
                           Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
        return Compose(transform_list)
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img,boxes

class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    def __call__(self, img, boxes=None):
        return self.to_tensor(img),boxes

class Normalize(object):
    def __init__(self, mean, std):
        self.norm = transforms.Normalize(mean,std)
    def __call__(self, tensor,boxes=None):
        return self.norm(tensor),boxes

class Resize(object):
    def __init__(self,size):
        self.resize = transforms.Scale(size, Image.BICUBIC)
        self.size = size
    def __call__(self, img, boxes=None):

        ## resize the bboxes
        np_image = np.array(img)
        img_h, img_w, _ = np_image.shape
        scale_h = self.size[1]/float(img_h)
        scale_w = self.size[0]/float(img_w)

        old_bbox_h = boxes[:,3]-boxes[:,1]
        old_bbox_w = boxes[:,2]-boxes[:,0]

        new_bbox_h = old_bbox_h*scale_h
        new_bbox_w = old_bbox_w*scale_w

        new_boxes = boxes.copy()
        new_boxes[:,0] = boxes[:,0]*scale_w
        new_boxes[:,1] = boxes[:,1]*scale_h
        new_boxes[:,2] = new_boxes[:,0] + new_bbox_w
        new_boxes[:,3] = new_boxes[:,1] + new_bbox_h

        ## resize the image
        new_img = self.resize(img)

        return new_img ,new_boxes

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def jaccard_boxa_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]

    return inter / area_a  # [A,B]

class ObjectCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form

    Return:
        (img, boxes, classes)
            img (Image): the cropped image

    """
    def __init__(self):
        self.extend_scale = 0.5
        ### extend crop area to X scale respect to original bboxes

    def __call__(self, image, boxes=None):
        np_image = np.array(image)
        img_height, img_width, _ = np_image.shape
        num_bboxes = len(boxes)
        chioce = random.randint(0,num_bboxes)
        box = boxes[chioce]
        left,top,right,bottom = box
        extend_w = (right - left)*self.extend_scale
        extend_h = (bottom - top)*self.extend_scale
        left = np.maximum(0.0,left-extend_w)
        top = np.maximum(0.0,top - extend_h)
        right = np.minimum(img_width,right + extend_w)
        bottom = np.minimum(img_height,bottom + extend_h)
        rect = np.array([int(left), int(top), int(right), int(bottom)])
        np_image = np_image[rect[1]:rect[3], rect[0]:rect[2],:]
        # cv2.imshow("test",np_image)
        # cv2.waitKey(10)
        pil_image = Image.fromarray(np_image)

        return pil_image


class RandomSampleCrop_FixedSize(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image

    """
    def __init__(self,size):
        self.size = size
    def __call__(self, image, boxes=None):
        image = np.array(image)
        height, width, _ = image.shape

        min_iou, max_iou = (0.9,float('inf'))

        # max trails (50)
        while True:
            current_image = image

            w,h = self.size


            left = random.uniform(width - w)
            top = random.uniform(height - h)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left+w), int(top+h)])

            # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
            # cv2.rectangle(image,(int(boxes[0,0]),int(boxes[0,1])),(int(boxes[0,2]),int(boxes[0,3])),color=(0,255,255))
            # cv2.imshow(",,",image)
            # cv2.waitKey(0)
            overlap = jaccard_boxa_numpy(boxes, rect)
            # is min and max overlap constraint satisfied? if not try again
            # if overlap.min() < min_iou and max_iou < overlap.max():
            #     continue
            if overlap.min() < min_iou:
                continue
            # cut the crop from the image
            current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                          :]


            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

            # mask in all gt boxes that above and to the left of centers
            m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

            # mask in all gt boxes that under and to the right of centers
            m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

            # mask in that both m1 and m2 are true
            mask = m1 * m2

            # have any valid boxes? try again if not
            if not mask.any():
                continue

            # take only matching gt boxes
            current_boxes = boxes[mask, :].copy()


            # should we use the box left and top corner or the crop's
            current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                              rect[:2])
            # adjust to crop (by substracting crop's left,top)
            current_boxes[:, :2] -= rect[:2]

            current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                              rect[2:])
            # adjust to crop (by substracting crop's left,top)
            current_boxes[:, 2:] -= rect[:2]

            # cv2.rectangle(current_image,(int(current_boxes[0,0]),int(current_boxes[0,1])),(int(current_boxes[0,2]),int(current_boxes[0,3])),(0,255,0))
            # cv2.imshow("test",current_image)
            # cv2.waitKey(10)
            current_image = Image.fromarray(current_image)
            # print(mode)
            # print(overlap.min())
            return current_image,current_boxes



class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image

    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.01, None),
            (0.03, None),
            (0.05, None),
            (0.07, None),
            (0.09, None),
            # # randomly sample a patch
            # (None, None),
        )

    def __call__(self, image, boxes=None):
        image = np.array(image)
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                image = Image.fromarray(image)
                return image

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(400):
                current_image = image

                w = random.uniform(0.01*width, width*0.5)
                h = random.uniform(0.01*height, height*0.5)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                # cv2.rectangle(image,(int(boxes[0,0]),int(boxes[0,1])),(int(boxes[0,2]),int(boxes[0,3])),color=(0,255,255))
                # cv2.imshow(",,",image)
                # cv2.waitKey(0)
                overlap = jaccard_numpy(boxes, rect)
                # is min and max overlap constraint satisfied? if not try again
                # if overlap.min() < min_iou and max_iou < overlap.max():
                #     continue
                if overlap.min() < min_iou:
                    continue
                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # # keep overlap with gt box IF center in sampled patch
                # centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                #
                # # mask in all gt boxes that above and to the left of centers
                # m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                #
                # # mask in all gt boxes that under and to the right of centers
                # m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                #
                # # mask in that both m1 and m2 are true
                # mask = m1 * m2
                #
                # # have any valid boxes? try again if not
                # if not mask.any():
                #     continue

                current_image = Image.fromarray(current_image)
                # print(mode)
                # print(overlap.min())
                return current_image