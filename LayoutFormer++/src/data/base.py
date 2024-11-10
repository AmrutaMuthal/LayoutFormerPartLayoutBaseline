import os.path as osp
from typing import List, Set, Union

import torch
from torch.utils.data import Dataset
import seaborn as sns

from utils import os_utils
from .load import load_publaynet_data, load_rico_data, load_partlayout_data


class LayoutDataset(Dataset):

    _label2index = None
    _index2label = None
    _colors = None

    split_file_names = ['train.pt', 'val.pt', 'test.pt']

    def __init__(self,
                 root: str,
                 data_name: str,
                 split: str,
                 max_num_elements: int,
                 label_set: Union[List, Set],
                 online_process: bool = True):

        self.root = f'{root}/{data_name}/'
        self.raw_dir = osp.join(self.root, 'raw')
        print(self.raw_dir)
        self.max_num_elements = max_num_elements
        self.label_set = label_set
        self.pre_processed_dir = osp.join(
            self.root, 'pre_processed_{}_{}'.format(self.max_num_elements, len(self.label_set)))
        assert split in ['train', 'val', 'test']

        if os_utils.files_exist(self.pre_processed_paths):
            idx = self.split_file_names.index('{}.pt'.format(split))
            print(f'Loading {split}...')
            self.data = torch.load(self.pre_processed_paths[idx])
        else:
            print(f'Pre-processing and loading {split}...')
            os_utils.makedirs(self.pre_processed_dir)
            split_dataset = self.load_raw_data()
            self.save_split_dataset(split_dataset)
            idx = self.split_file_names.index('{}.pt'.format(split))
            self.data = torch.load(self.pre_processed_paths[idx])

        self.online_process = online_process
        if not self.online_process:
            self.data = [self.process(item) for item in self.data]

    @property
    def pre_processed_paths(self):
        return [
            osp.join(self.pre_processed_dir, f) for f in self.split_file_names
        ]

    @classmethod
    def label2index(self, label_set):
        if self._label2index is None:
            self._label2index = dict()
            for idx, label in enumerate(label_set):
                self._label2index[label] = idx + 1
        return self._label2index

    @classmethod
    def index2label(self, label_set):
        if self._index2label is None:
            self._index2label = dict()
            for idx, label in enumerate(label_set):
                self._index2label[idx + 1] = label
        return self._index2label

    @property
    def colors(self):
        if self._colors is None:
            n_colors = len(self.label_set) + 1
            colors = sns.color_palette('husl', n_colors=n_colors)
            self._colors = [
                tuple(map(lambda x: int(x * 255), c)) for c in colors
            ]
        return self._colors

    def save_split_dataset(self, split_dataset):
        torch.save(split_dataset[0], self.pre_processed_paths[0])
        torch.save(split_dataset[1], self.pre_processed_paths[1])
        torch.save(split_dataset[2], self.pre_processed_paths[2])

    def load_raw_data(self) -> list:
        raise NotImplementedError

    def process(self, data) -> dict:
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.online_process:
            sample = self.process(self.data[idx])
        else:
            sample = self.data[idx]
        return sample


class PubLayNetDataset(LayoutDataset):
    labels = [
        'text',
        'title',
        'list',
        'table',
        'figure',
    ]

    def __init__(self,
                 root: str,
                 split: str,
                 max_num_elements: int,
                 online_process: bool = True):
        data_name = 'publaynet'
        super().__init__(root,
                         data_name,
                         split,
                         max_num_elements,
                         label_set=self.labels,
                         online_process=online_process)

    def load_raw_data(self) -> list:
        return load_publaynet_data(self.raw_dir, self.max_num_elements,
                                   self.label_set, self.label2index(self.label_set))


class RicoDataset(LayoutDataset):

    labels = [
        'Text', 'Image', 'Icon', 'List Item', 'Text Button', 'Toolbar',
        'Web View', 'Input', 'Card', 'Advertisement', 'Background Image',
        'Drawer', 'Radio Button', 'Checkbox', 'Multi-Tab', 'Pager Indicator',
        'Modal', 'On/Off Switch', 'Slider', 'Map View', 'Button Bar', 'Video',
        'Bottom Navigation', 'Number Stepper', 'Date Picker'
    ]

    def __init__(self,
                 root: str,
                 split: str,
                 max_num_elements: int,
                 online_process: bool = True):
        data_name = 'rico'
        super().__init__(root,
                         data_name,
                         split,
                         max_num_elements,
                         label_set=self.labels,
                         online_process=online_process)

    def load_raw_data(self) -> list:
        return load_rico_data(self.raw_dir, self.max_num_elements,
                              self.label_set, self.label2index(self.label_set))

class PartLayoutDataset(LayoutDataset):
    id_to_label = {
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

    labels = list(id_to_label.values())

    def __init__(self,
                 root: str,
                 split: str,
                 max_num_elements: int,
                 online_process: bool = True):
        data_name = 'partlayout'
        super().__init__(root,
                         data_name,
                         split,
                         max_num_elements,
                         label_set=self.labels,
                         online_process=online_process)

    def load_raw_data(self) -> list:
        return load_partlayout_data(self.raw_dir, self.max_num_elements,
                              self.label_set, self.label2index(self.label_set))
