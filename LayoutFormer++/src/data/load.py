import json
from pathlib import Path
from typing import List, Set, Union, Dict

import torch
from pycocotools.coco import COCO
"""
load raw data
"""

PART_ID_TO_LABEL = {
      0 : 'unlabeled',
      1 : 'cow head',
      2 : 'cow left horn',
      3 : 'cow right horn',
      4 : 'cow torso',
      5 : 'cow neck',
      6 : 'cow left front upper leg',
      7 : 'cow left front lower leg',
      8 : 'cow right front upper leg',
      9 : 'cow right front lower leg',
      10 : 'cow left back upper leg',
      11 : 'cow left back lower leg',
      12 : 'cow right back upper leg',
      13 : 'cow right back lower leg',
      14 : 'cow tail',
      15 : 'sheep head',
      16 : 'sheep left horn',
      17 : 'sheep right horn',
      18 : 'sheep torso',
      19 : 'sheep neck',
      20 : 'sheep left front upper leg',
      21 : 'sheep left front lower leg',
      22 : 'sheep right front upper leg',
      23 : 'sheep right front lower leg',
      24 : 'sheep left back upper leg',
      25 : 'sheep left back lower leg',
      26 : 'sheep right back upper leg',
      27 : 'sheep right back lower leg',
      28 : 'sheep tail',
      29 : 'bird head',
      30 : 'bird torso',
      31 : 'bird neck',
      32 : 'bird left wing',
      33 : 'bird right wing',
      34 : 'bird left leg',
      35 : 'bird left foot',
      36 : 'bird right leg',
      37 : 'bird right foot',
      38 : 'bird tail',
      39 : 'person head',
      40 : 'person torso',
      41 : 'person neck',
      42 : 'person left lower arm',
      43 : 'person left upper arm',
      44 : 'person left hand',
      45 : 'person right lower arm',
      46 : 'person right upper arm',
      47 : 'person right hand',
      48 : 'person left lower leg',
      49 : 'person left upper leg',
      50 : 'person left foot',
      51 : 'person right lower leg',
      52 : 'person right upper leg',
      53 : 'person right foot',
      54 : 'cat head',
      55 : 'cat torso',
      56 : 'cat neck',
      57 : 'cat left front leg',
      58 : 'cat left front paw',
      59 : 'cat right front leg',
      60 : 'cat right front paw',
      61 : 'cat left back leg',
      62 : 'cat left back paw',
      63 : 'cat right back leg',
      64 : 'cat right back paw',
      65 : 'cat tail',
      66 : 'dog head',
      67 : 'dog torso',
      68 : 'dog neck',
      69 : 'dog left front leg',
      70 : 'dog left front paw',
      71 : 'dog right front leg',
      72 : 'dog right front paw',
      73 : 'dog left back leg',
      74 : 'dog left back paw',
      75 : 'dog right back leg',
      76 : 'dog right back paw',
      77 : 'dog tail',
      78 : 'dog muzzle',
      79 : 'horse head',
      80 : 'horse left front hoof',
      81 : 'horse right front hoof',
      82 : 'horse torso',
      83 : 'horse neck',
      84 : 'horse left front upper leg',
      85 : 'horse left front lower leg',
      86 : 'horse right front upper leg',
      87 : 'horse right front lower leg',
      88 : 'horse left back upper leg',
      89 : 'horse left back lower leg',
      90 : 'horse right back upper leg',
      91 : 'horse right back lower leg',
      92 : 'horse tail',
      93 : 'horse left back hoof',
      94 : 'horse right back hoof'
  }

def load_publaynet_data(raw_dir: str, max_num_elements: int,
                        label_set: Union[List, Set], label2index: Dict):

    def is_valid(element):
        label = coco.cats[element['category_id']]['name']
        if label not in set(label_set):
            return False
        x1, y1, width, height = element['bbox']
        x2, y2 = x1 + width, y1 + height

        if x1 < 0 or y1 < 0 or W < x2 or H < y2:
            return False
        if x2 <= x1 or y2 <= y1:
            return False

        return True

    train_list, val_list = None, None
    raw_dir = Path(raw_dir) / 'publaynet'
    for split_publaynet in ['train', 'val']:
        dataset = []
        coco = COCO(raw_dir / f'{split_publaynet}.json')
        for img_id in sorted(coco.getImgIds()):
            ann_img = coco.loadImgs(img_id)
            W = float(ann_img[0]['width'])
            H = float(ann_img[0]['height'])
            name = ann_img[0]['file_name']
            if H < W:
                continue

            elements = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
            _elements = list(filter(is_valid, elements))
            filtered = len(elements) != len(_elements)
            elements = _elements

            N = len(elements)
            if N == 0 or max_num_elements < N:
                continue

            bboxes = []
            labels = []

            for element in elements:
                # bbox
                x1, y1, width, height = element['bbox']
                b = [x1 / W, y1 / H,  width / W, height / H]  # bbox format: ltwh
                bboxes.append(b)

                # label
                label = coco.cats[element['category_id']]['name']
                labels.append(label2index[label])

            bboxes = torch.tensor(bboxes, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)

            data = {
                'name': name,
                'bboxes': bboxes,
                'labels': labels,
                'canvas_size': [W, H],
                'filtered': filtered,
            }
            dataset.append(data)

        if split_publaynet == 'train':
            train_list = dataset
        else:
            val_list = dataset

    # shuffle train with seed
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(train_list), generator=generator)
    train_list = [train_list[i] for i in indices]

    # train_list -> train 95% / val 5%
    # val_list -> test 100%

    s = int(len(train_list) * .95)
    train_set = train_list[:s]
    test_set = train_list[s:]
    val_set = val_list
    split_dataset = [train_set, test_set, val_set]
    return split_dataset


def load_rico_data(raw_dir: str, max_num_elements: int,
                   label_set: Union[List, Set], label2index: Dict):

    def is_valid(element):
        if element['componentLabel'] not in set(label_set):
            return False
        x1, y1, x2, y2 = element['bounds']
        if x1 < 0 or y1 < 0 or W < x2 or H < y2:
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        return True

    def append_child(element, elements):
        if 'children' in element.keys():
            for child in element['children']:
                elements.append(child)
                elements = append_child(child, elements)
        return elements

    dataset = []
    raw_dir = Path(raw_dir) / 'semantic_annotations'
    for json_path in sorted(raw_dir.glob('*.json')):
        with json_path.open() as f:
            ann = json.load(f)

        B = ann['bounds']
        W, H = float(B[2]), float(B[3])
        if B[0] != 0 or B[1] != 0 or H < W:
            continue

        elements = append_child(ann, [])
        _elements = list(filter(is_valid, elements))
        filtered = len(elements) != len(_elements)
        elements = _elements

        N = len(elements)
        if N == 0 or N > max_num_elements:
            continue

        bboxes = []
        labels = []

        for element in elements:
            x1, y1, x2, y2 = element['bounds']
            b = [x1 / W, y1 / H, (x2-x1) / W, (y2-y1) / H]  # bbox format: ltwh
            bboxes.append(b)

            label = label2index[element['componentLabel']]
            labels.append(label)

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        data = {
            'name': json_path.name,
            'bboxes': bboxes,
            'labels': labels,
            'canvas_size': [W, H],
            'filtered': filtered,
        }
        dataset.append(data)

    # shuffle with seed
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(dataset), generator=generator)
    dataset = [dataset[i] for i in indices]

    # train 85% / val 5% / test 10%
    N = len(dataset)
    s = [int(N * .85), int(N * .90)]
    train_set = dataset[:s[0]]
    test_set = dataset[s[0]:s[1]]
    val_set = dataset[s[1]:]
    split_dataset = [train_set, test_set, val_set]

    return split_dataset

def load_partlayout_data(raw_dir: str, max_num_elements: int,
                        label_set: Union[List, Set], label2index: Dict):
    def is_valid(element):
        label = PART_ID_TO_LABEL[element['category_id']]
        if label not in set(label_set):
            return False
        x1, y1 = element['x_min'], element['y_min']
        x2, y2 = element['x_max'], element['y_max']

        if x1 < 0 or y1 < 0:
            return False
        if x2 <= x1 or y2 <= y1:
            return False

        return True

    train_list, val_list = None, None
    # raw_dir = Path(raw_dir)
    for split_partlayout in ['train', 'test', 'val']:
        dataset = []
        json_path = Path(raw_dir) / f'layout_{split_partlayout}.json'
        with json_path.open() as f:
            ann = json.load(f)
            if type(ann) == str:
                ann = json.loads(ann)
        for idx, layout in enumerate(ann):

            H, W = layout['height'], layout['width']

            elements = layout['children']
            _elements = list(filter(is_valid, elements))
            filtered = len(elements) != len(_elements)
            elements = _elements

            N = len(elements)
            if N == 0 or max_num_elements < N:
                continue

            bboxes = []
            labels = []

            for element in elements:
                # bbox
                x1, y1 = element['x_min'], element['y_min']
                width, height = element['width'], element['height']
                b = [x1 , y1,  width, height]  # bbox format: ltwh
                bboxes.append(b)

                # label
                # label = [element['category_id']]['name']
                labels.append(element['category_id'])

            bboxes = torch.tensor(bboxes, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)

            data = {
                'name': idx,
                'bboxes': bboxes,
                'labels': labels,
                'canvas_size': [512, 512],
                'filtered': filtered,
            }
            dataset.append(data)

        if split_partlayout == 'train':
            train_list = dataset
        
        elif split_partlayout == 'test':
            test_list = dataset

        else:
            val_list = dataset

    # shuffle train with seed
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(train_list), generator=generator)
    train_list = [train_list[i] for i in indices]


    train_set = train_list
    test_set = test_list
    val_set = val_list
    split_dataset = [train_set, test_set, val_set]
    return split_dataset
