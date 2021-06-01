import torch
import numpy as np
from .metrics import ap_per_class

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def coco_map(all_pred_boxes, all_gt_boxes, num_classes, plot=False, save_dir='.'):
    """
    Args:
        all_pred_boxes: list of tensors [tensor[_, 6]], content is [x0,y0,x1,y1,cls,conf]
        all_gt_boxes: list of tensors[tensor[_,5]], content is [x0,y0,x1,y1,cls]
        num_classes: number of classes
    """
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    stats = []
    # for each image
    for img_i in range(len(all_pred_boxes)):
        if img_i % 100 == 0:
            print('img_i: ', img_i)
        pred_boxes = torch.tensor(all_pred_boxes[img_i][:, :4])
        pred_cls = torch.tensor(all_pred_boxes[img_i][:, 4])
        pred_conf = torch.tensor(all_pred_boxes[img_i][:, 5])
        gt_boxes = torch.tensor(all_gt_boxes[img_i][:, :4])
        gt_cls = torch.tensor(all_gt_boxes[img_i][:, 4])

        correct = torch.zeros(pred_boxes.shape[0], niou, dtype=torch.bool)
        nl = len(gt_boxes)
        # for each class
        for c in range(num_classes):
            ti = torch.nonzero(gt_cls == c, as_tuple=False).view(-1)  # prediction indices
            pi = torch.nonzero(pred_cls == c, as_tuple=False).view(-1)  # target indices

            if pi.shape[0]:
                ious, indices = box_iou(pred_boxes[pi], gt_boxes[ti]).max(1)
                
                detected_set = set()
                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                    d = ti[indices[j]]  # detected target
                    if d.item() not in detected_set:
                        detected_set.add(d.item())
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected_set) == nl: # all targets already located in image
                            break
        
        stats.append((correct, pred_conf, pred_cls, gt_cls))
    
    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plot, save_dir=save_dir,)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    print('precision: ', mp)
    print('recall:    ', mr)
    print('mAP50:     ', map50)
    print('mAP:       ', map)
    return mp, mr, map50, map