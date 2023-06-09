import argparse
import logging
import os
import cv2
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import torch.utils.data

from hardware.camera import RealSenseCamera

from model.utils.config import cfg, read_cfgs
from model.MGN import MGN
from model.utils.data_viewer import dataViewer
from model.utils.net_utils import objgrasp_inference
from model.utils.blob import image_unnormalize
from model.utils.blob import prep_im_for_blob, image_normalize

logging.basicConfig(level=logging.INFO)

def init_network():

    conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
    Network = MGN(32, class_agnostic=True, feat_name=args.net,         
                      feat_list=('conv' + conv_num,), pretrained=True)
    Network.create_architecture()

    # Loading weight                                                              
    if args.resume:                                                                   # 학습할 때, 이전에 저장한 체크포인트 파일을 불러와서 모델의 가중치를 복원하는 역할
        output_dir = 'output/vmrdcompv1/res101'         
        load_name = os.path.join(output_dir,
                                 args.frame + '_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                     args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        Network.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.RCNN_COMMON.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    Network.cuda()
    
    return Network

def move_to_cuda(data_batch):
    cuda_data_batch = []
    for data in data_batch:
        # 만약 데이터가 PyTorch 텐서가 아니라면 텐서로 변환합니다.
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        cuda_data_batch.append(data.to('cuda'))
    return tuple(cuda_data_batch)

def prep_im_for_blob(im, target_size, max_size, fix_size = False):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    im_shape = im.shape
    im_scale = {}
    if not fix_size:
        im_size_min = np.min(im_shape[0:2])
        im_scale['x'] = float(target_size) / float(im_size_min)
        im_scale['y'] = float(target_size) / float(im_size_min)
    else:
        im_size_y, im_size_x = im.shape[:2]
        im_scale['x'] = float(target_size) / float(im_size_x)
        im_scale['y'] = float(target_size) / float(im_size_y)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = cv2.resize(im, None, None, fx=im_scale['x'], fy=im_scale['y'],
                    interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def create_data_batch(rgb_image):
    #im_data = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0).to('cuda')
    im_data = rgb_image

    # 높이, 너비, 높이 스케일, 너비 스케일을 계산합니다.
    height, width = rgb_image.shape[:2]
    random_scale_ind = np.random.randint(0, high=len(cfg.SCALES))
    im_data, im_scale = prep_im_for_blob(im_data, cfg.SCALES[random_scale_ind], cfg.TRAIN.COMMON.MAX_SIZE, fix_size=False)
    height_scale, width_scale = im_scale['y'], im_scale['x']
    im_info = torch.tensor([height, width, height_scale, width_scale]).unsqueeze(0).to('cuda')

    pixel_means = cfg.PIXEL_MEANS if cfg.PRETRAIN_TYPE == "pytorch" else cfg.PIXEL_MEANS_CAFFE
    pixel_stds = cfg.PIXEL_STDS if cfg.PRETRAIN_TYPE == "pytorch" else np.array([[[1., 1., 1.]]])
    im_data = image_normalize(im_data, mean=pixel_means, std=pixel_stds)

    im_data = torch.from_numpy(im_data).float().permute(2, 0, 1).unsqueeze(0).to('cuda')

    # gt_boxes, gt_grasps, num_boxes, num_grasps, gt_grasp_inds를 생성합니다.
    gt_boxes = torch.tensor([[1., 1., 1., 1., 1.]], device='cuda:0')
    gt_grasps = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]], device='cuda:0')
    num_boxes = torch.tensor([0], device='cuda:0')
    num_grasps = torch.tensor([0], device='cuda:0')
    gt_grasp_inds = torch.tensor([[0]], device='cuda:0')
    
    # 생성된 모든 요소를 data_batch로 그룹화합니다.
    data_batch = (im_data, im_info, gt_boxes, gt_grasps, num_boxes, num_grasps, gt_grasp_inds)
    
    return data_batch

def draw_graspdet_with_owner(o_dets, g_dets, g_inds):
    """
    :param im: original image numpy array
    :param o_dets: object detections. size N x 5 with 4-d bbox and 1-d class
    :param g_dets: grasp detections. size N x 8 numpy array
    :param g_inds: grasp indice. size N numpy array
    :return:
    """
    if o_dets.shape[0] > 0:
        o_inds = np.arange(o_dets.shape[0])
        object_data = o_dets[(o_dets[:,:4].sum(-1)) > 0].astype(int)
        g_inds = g_inds
        grasp_data = g_dets[(g_dets[:,:8].sum(-1)) > 0].astype(int)
    return o_inds, object_data, g_inds, grasp_data

def vis_gt(data_list):
    im_vis = image_unnormalize(data_list[0].permute(1, 2, 0).cpu().numpy())
    im_vis = cv2.resize(im_vis, None, None, fx=1. / data_list[1][3].item(), fy=1. / data_list[1][2].item(),
                            interpolation=cv2.INTER_LINEAR)
    o_inds, object_data, g_inds, grasp_data = draw_graspdet_with_owner(data_list[2].cpu().numpy(),
                                    data_list[3].cpu().numpy(), data_list[6].cpu().numpy())
    return im_vis, o_inds, object_data, g_inds, grasp_data

def detection_filter(all_boxes, all_grasp=None, max_per_image=10):
    # Limit to max_per_image detections *over all classes*
    image_scores = np.hstack([all_boxes[j][:, -1]
                              for j in range(1, len(all_boxes))])
    if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, len(all_boxes)):
            keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
            all_boxes[j] = all_boxes[j][keep, :]
            if all_grasp is not None:
                all_grasp[j] = all_grasp[j][keep, :]
    if all_grasp is not None:
        return all_boxes, all_grasp
    else:
        return all_boxes

if __name__ == '__main__':
    #parse arguments
    args = read_cfgs()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id=147122072422)
    cam.connect() #카메라 연결
    logging.info('Done')

    # Load Network
    logging.info('Loading model...')
    Network = init_network()
    Network.eval()
    logging.info('Done')

    # Evaluate Network
    try:
        fig = plt.figure(figsize=(15,7))
        i = 0
        while i <= 30:
            i += 1
            image_bundle = cam.get_image_bundle()
            rgb_image = image_bundle['rgb']
            data_batch = create_data_batch(rgb_image)

            data_batch = move_to_cuda(data_batch)

            # Forward process
            with torch.no_grad():
                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, rois_label, grasp_loc, grasp_prob, \
            grasp_bbox_loss, grasp_cls_loss, grasp_conf_label, grasp_all_anchors = Network(data_batch)
            boxes = rois[:, :, 1:5]

            # Collect results
            max_per_image = 5
            det_box, det_grasps = objgrasp_inference(cls_prob[0].data,
                                                     bbox_pred[0].data if bbox_pred is not None else bbox_pred,
                                                     grasp_prob.data, grasp_loc.data, data_batch[1][0].data,
                                                     boxes[0].data,
                                                     class_agnostic=True,
                                                     g_box_prior=grasp_all_anchors.data, for_vis=False, topN_g=1)
            det_box, det_grasps = detection_filter(det_box, det_grasps, max_per_image)

            det_grasps[1] = det_grasps[1].reshape(det_grasps[1].shape[0], 1, det_grasps[1].shape[1])

            if det_box[1].shape[0] > 0:
                g_inds = torch.Tensor(np.arange(det_box[1].shape[0])).unsqueeze(1).repeat(1, 1)
            else:
                g_inds = torch.Tensor([])

            data_list = [data_batch[0][0], data_batch[1][0], torch.Tensor(det_box[1]),
                        torch.Tensor(det_grasps[1]).view(-1, det_grasps[1].shape[-1]), None, None,
                        g_inds.long().view(-1)]
            
            im_vis, o_inds, object_data, g_inds, grasp_data = vis_gt(data_list)

            plt.ion()
            plt.clf()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(rgb_image)

            # object bounding box를 그립니다.
            for box in object_data:
                x1, y1, x2, y2, _ = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)

            # grasp bounding box를 그립니다.
            for grasp in grasp_data:
                x1, y1, x2, y2, x3, y3, x4, y4 = grasp
                vertical_line1 = Line2D([x1, x2], [y1, y2], linewidth=2, color='r')
                vertical_line2 = Line2D([x3, x4], [y3, y4], linewidth=2, color='r')
                horizontal_line1 = Line2D([x2, x3], [y2, y3], linewidth=1, color='b')
                horizontal_line2 = Line2D([x1, x4], [y1, y4], linewidth=1, color='b')
                ax.add_line(vertical_line1)
                ax.add_line(vertical_line2)
                ax.add_line(horizontal_line1)
                ax.add_line(horizontal_line2)
            
            ax.axis('off')
            plt.pause(0.1)
            fig.canvas.draw()
            plt.show()
    finally: #중간에 오류가나도 무조건 실행
        pass


# qt error가 나면 opencv-python 버전을 최소 4.2.0.34 까지 낮춰야 돌아간다. (pip install opencv-python==4.2.0.34)