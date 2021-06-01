
from coco_map import coco_map
from pathlib import Path
from PIL import Image
import numpy as np

def read_txt(f, width, height):
    d = np.loadtxt(str(f))
    if len(d) == 0:
        return None
    if len(d.shape) == 1:
        d = d[None]
    scale = np.array([width, height, width, height])
    c = d[:, 0]
    xywh = d[:, 1:5] * scale
    cx, cy, w, h = xywh.transpose()
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = x0 + w, y0 + h
    boxes = np.stack([x0, y0, x1, y1], 1)
    res = {
        'classes': c,
        'boxes': boxes,
    }
    if d.shape[1] == 6:
        res['scores'] = d[:, 5]
    return res

def read_data(image_dir, pred_dir):
    image_dir = Path(image_dir)
    pred_dir = Path(pred_dir)
    image_files = list(image_dir.glob('**/*.jpg'))
    label_files = [str(x).replace('images', 'labels').replace('.jpg', '.txt') for x in image_files]
    proposal_files = [pred_dir / f'{x.stem}.txt' for x in image_files]
    data_dict = {}
    for i in range(len(image_files)):
        image_file = image_files[i]
        img = Image.open(str(image_file))
        width, height = img.size
        
        if not Path(proposal_files[i]).is_file() or not Path(label_files[i]).is_file():
            print('Warning: %s or %s is not a file or does not exist' % (proposal_files[i], label_files[i]))
            continue
        gt_labels = read_txt(label_files[i], width, height)                
        pred_labels = read_txt(proposal_files[i], width, height)
        data_dict[image_file.stem] = {'pred_labels':pred_labels, 'gt_labels':gt_labels}
    return data_dict

if __name__ == '__main__':
    image_dir = '/home/luojiapeng/Projects/TrafficSignDet/data/TT-100K/yolo_data_1cls/val/images'
    pred_dir = '/home/luojiapeng/Projects/TrafficSignDet/yolov5/runs/test/yolov5t-960-val-1cls/labels'

    data_dict = read_data(image_dir, pred_dir)

    gt_list = []
    pred_list = []
    for k, v in data_dict.items():
        gt = v['gt_labels']
        pred = v['pred_labels']
        gt_list.append(np.concatenate([gt['boxes'], gt['classes'].reshape([-1, 1])], 1))
        pred_list.append(np.concatenate([pred['boxes'], pred['classes'].reshape([-1, 1]), pred['scores'].reshape([-1, 1])], 1))
    coco_map(pred_list, gt_list, 1)
