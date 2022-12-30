# Adapted from https://github.com/zllrunning/face-parsing.PyTorch
#!/usr/bin/python
# -*- encoding: utf-8 -*-

from rumpy.sr_tools.face_segmentation.models import BiSeNet

import torch

import sys
import click
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import glob
from tqdm import tqdm
from pathlib import Path
import rumpy.shared_framework.configuration.constants as sconst

# Colors for all 20 parts
part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],  # fill in the classes here when known...
               [255, 0, 85], [255, 0, 170],  #
               [0, 255, 0], [85, 255, 0], [170, 255, 0],  #
               [0, 255, 85], [0, 255, 170],  #
               [0, 0, 255], [85, 0, 255], [170, 0, 255],  #
               [0, 85, 255], [0, 170, 255],  #
               [255, 255, 0], [255, 255, 85], [255, 255, 170],  #
               [255, 0, 255], [255, 85, 255], [255, 170, 255],  #
               [0, 255, 255], [85, 255, 255], [170, 255, 255]]  #


def vis_parsing_maps(im, parsing_anno, stride, save_path, orig_dim, im_name, save_im=False):

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.5, vis_parsing_anno_color, 0.5, 0)

    vis_parsing_anno_color = cv2.resize(vis_parsing_anno_color, orig_dim, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_path, im_name), vis_parsing_anno_color)

    if save_im:
        base, ext = im_name.split('.')
        vis_im = cv2.resize(vis_im, orig_dim, interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(os.path.join(save_path, base+'_superimposed.'+ext), vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


@click.command()
@click.option('--in_dir', default=os.path.join(sconst.data_directory, 'celeba/png_samples/celeba_png_align/eval_samples'),
              help='Input directory/file for face segmentation.')
@click.option("--save_superimposed_images", is_flag=True, help='Set this flag to additionally save '
                                                               'images with superimposed segmentation map')
@click.option('--gpu', is_flag=True, help='Set this flag to accelerate processing via GPU.')
@click.option('--weights_path', default='sr_tools/face_segmentation/weights.pth',
              help='location of pre-trained weights for BiSeNet')
def segment(in_dir, save_superimposed_images, gpu, weights_path):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    if gpu:
        map_location = None
        net.cuda()
    else:
        map_location = torch.device('cpu')

    net.load_state_dict(torch.load(weights_path, map_location=map_location))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    filenames = []

    if os.path.isdir(in_dir):
        for extension in ['*.jpg', '*.png', '*.bmp']:
            filenames.extend(glob.glob(os.path.join(in_dir, extension)))
        filenames.sort()
        out_loc = os.path.join(in_dir, 'segmentation_patterns')
    else:
        filenames = [in_dir]
        out_loc = os.path.join(Path(in_dir).parent, 'segmentation_patterns')

    if not os.path.exists(out_loc):
        os.mkdir(out_loc)

    with torch.no_grad():
        for image_path in tqdm(filenames):
            img = Image.open(image_path)
            orig_dim = (img.width, img.height)
            if img.mode == 'RGBA':  # just in case images have transparency channel included
                img = img.convert('RGB')
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            if gpu:
                img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=save_superimposed_images, orig_dim=orig_dim,
                             save_path=out_loc, im_name=image_path.split('/')[-1])


if __name__ == "__main__":
    segment(sys.argv[1:])  # for use when debugging with pycharm


