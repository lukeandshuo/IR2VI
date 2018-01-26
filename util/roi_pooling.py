import torch
import torch.nn as nn
import torch.autograd as ag
import math
from PIL import Image
from torch.autograd.function import Function
from torch._thnn import type2backend
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d,adaptive_max_pool2d



def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):
    '''

    :param input: input feature or images (batch_size,channel,height,width)
    :param rois: cropped bboxing (batch_size,bbox), bbox=[x1,y1,x2,y2]
    :param size: output size (height,width)
    :param spatial_scale:  down scale bboxes
    :return: (batch_size,channel,cropped_height,cropped_width)
    '''
    assert (rois.dim() == 2)
    assert (rois.size(1) == 4)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois.mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        # im_idx = roi[0]
        # im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)] for multiple rois
        im = input[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
        output.append(adaptive_avg_pool2d(im, size))
    return torch.cat(output, 0)


if __name__ == '__main__':
    img = Image.open('../imgs/sensiac.png')
    # img.show()
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_variable = ag.Variable(img_tensor,requires_grad=True)
    input = img_variable.cuda()
    rois = ag.Variable(torch.LongTensor([[ 3, 1, 400, 300]]), requires_grad=False)
    # rois = ag.Variable(torch.LongTensor([[0,3,3,8,8]]),requires_grad=False)

    # out = adaptive_max_pool(input, (7, 7))
    # out.backward(out.data.clone().uniform_())
    input = input.cuda()
    rois = rois.cuda()
    out = roi_pooling(input, rois, size=(128, 128))
    out.backward(out.data.clone().uniform_())
    out_img = out.cpu().data
    out_img = torch.squeeze(out_img,0)
    out_img = transforms.ToPILImage()(out_img)
    out_img.show()
    print(out_img)
