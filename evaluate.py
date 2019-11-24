import cv2
import numpy as np
import torch
import time
import os
import sys
import argparse
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks import get_network
from data import get_loader
import torchvision.transforms as std_trnsf
from utils import joint_transforms as jnt_trnsf
from utils.metrics import MultiThresholdMeasures

def str2bool(s):
    return s.lower() in ('t', 'true', 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', help='path to ckpt file',type=str,
            default='./models/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth')
    parser.add_argument('--dataset', type=str, default='figaro',
            help='Name of dataset you want to use default is "figaro"')
    parser.add_argument('--data_dir', help='path to Figaro1k folder', type=str, default='./data/Figaro1k')
    parser.add_argument('--networks', help='name of neural network', type=str, default='pspnet_resnet101')
    parser.add_argument('--save_dir', default='./overlay',
            help='path to save overlay images, default=None and do not save images in this case')
    parser.add_argument('--use_gpu', type=str2bool, default=True,
            help='True if using gpu during inference')
    parser.add_argument('--visualize', type=str2bool, default=False,
            help='if set to True, show input image and output probability map')
    parser.add_argument('--inference', type=str2bool, default=False,
            help='if set to True, do not calculate accuracy or metric')

    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    img_dir = os.path.join(data_dir, 'Original', 'Testing')
    network = args.networks.lower()
    save_dir = args.save_dir
    device = 'cuda' if args.use_gpu else 'cpu'
    visualize = args.visualize
    if visualize:
        import matplotlib.pyplot as plt
    inference = args.inference

    assert os.path.exists(ckpt_dir)
    assert os.path.exists(data_dir)
    assert os.path.exists(os.path.split(save_dir)[0])

    if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # prepare network with trained parameters
    net = get_network(network).to(device)
    state = torch.load(ckpt_dir)
    net.load_state_dict(state['weight'])

    # this is the default setting for train_verbose.py
    # test_joint_transforms = jnt_trnsf.Compose([
    #     jnt_trnsf.Safe32Padding()
    # ])
    input_size = 224
    test_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.Resize(input_size)
    ])

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
        ])

    # transforms only on mask
    mask_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
        ])
    if not inference:
        test_loader = get_loader(dataset=args.dataset,
                                data_dir=data_dir,
                                train=False,
                                joint_transforms=test_joint_transforms,
                                image_transforms=test_image_transforms,
                                mask_transforms=mask_transforms,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8)

        # prepare measurements
        metric = MultiThresholdMeasures()
        metric.reset()
        durations = list()

        # prepare images
        imgs = [os.path.join(img_dir, k) for k in sorted(os.listdir(img_dir)) if k.endswith('.jpg')]
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                print('[{:3d}/{:3d}] processing image... '.format(i+1, len(test_loader)))
                net.eval()
                data, label = data.to(device), label.to(device)
                
                # inference
                start = time.time()
                logit = net(data)
                duration = time.time() - start

                # prepare mask
                data = data.cpu()[0].numpy().transpose(1, 2, 0)
                pred = logit.cpu()[0].numpy().transpose(1, 2, 0)
                pred = pred.reshape(pred.shape[0], pred.shape[1])
                mask = pred >= -0.3
                mh, mw = pred.shape
                if visualize:
                    plt.figure(0)
                    plt.subplot(1, 2, 1)
                    plt.imshow(data)
                    plt.subplot(1, 2, 2)
                    plt.imshow(pred, cmap='gray')
                    plt.show()
                
                mask_n = np.zeros((mh, mw, 3))
                mask_n[:,:,0] = 255
                mask_n[:,:,0] *= mask

                path = os.path.join(save_dir, "figaro_img_%04d.png" % i)
                image_n = cv2.imread(imgs[i])

                # addWeighted
                image_n = image_n * 0.5 +  cv2.resize(mask_n, (image_n.shape[1], image_n.shape[0])) * 0.5

                # log measurements
                metric.update((logit, label))
                durations.append(duration)

                # write overlay image
                cv2.imwrite(path,image_n)


        # compute measurements
        iou = metric.compute_iou()
        f = metric.compute_f1()
        acc = metric.compute_accuracy()
        avg_fps = len(durations)/sum(durations)

        print('Avg-FPS:', avg_fps)
        print('Pixel-acc:', acc)
        print('F1-score:', f)
        print('IOU:', iou)
    else:
        durations = []
        # prepare images
        imgs = [os.path.join(data_dir, k) for k in sorted(os.listdir(data_dir)) if k.endswith('.jpg')]
        img_data = [torch.from_numpy(np.array(Image.open(fname)).transpose(2, 0, 1)[np.newaxis, :, :, :]) for fname in imgs]
        with torch.no_grad():
            for i, data in enumerate(img_data):
                print('[{:3d}/{:3d}] processing image... '.format(i+1, len(imgs)))
                net.eval()
                data = data.to(device, dtype=torch.float32) / 255
                
                # inference
                start = time.time()
                logit = net(data)
                duration = time.time() - start

                # prepare mask
                data = data.cpu()[0].numpy().transpose(1, 2, 0)
                pred = logit.cpu()[0].numpy().transpose(1, 2, 0)
                pred = pred.reshape(pred.shape[0], pred.shape[1])
                mask = pred >= -0.3
                mh, mw = pred.shape
                
                
                mask_n = np.zeros((mh, mw, 3))
                mask_n[:,:,0] = 255
                mask_n[:,:,0] *= mask

                mask_r = np.zeros((mh, mw, 3))
                mask_r[:, :, 0] = np.maximum(pred, 0) * (255/(pred.max()))

                if visualize:
                    plt.figure(0)
                    plt.subplot(1, 3, 1)
                    plt.imshow(data)
                    plt.subplot(1, 3, 2)
                    plt.imshow(pred, cmap='gray')
                    plt.subplot(1, 3, 3)
                    plt.imshow(mask_r.astype('uint8'))
                    plt.show()

                path = os.path.join(save_dir, "img_%04d.png" % i)
                image_n = cv2.imread(imgs[i])

                # addWeighted
                image_n = image_n * 0.5 +  cv2.resize(mask_r, (image_n.shape[1], image_n.shape[0])) * 0.5

                # log measurements
                durations.append(duration)

                # write overlay image
                cv2.imwrite(path,image_n)


        # compute measurements
        avg_fps = len(durations)/sum(durations)
