import cv2
import json
import numpy as np
import random
import re
import torch
import torch.utils.data as data
from pathlib import Path
from torchvision import transforms

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def read_img(path):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

@DATASET_REGISTRY.register()
class CTUDVideoDataset(data.Dataset):
    """CTUD dataset for Training
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(CTUDVideoDataset, self).__init__()
        self.opt = opt
        self.scale = opt['scale']
        self.gt_size = opt['gt_size']
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']
        self.filename_tmpl = opt['filename_tmpl']
        self.filename_ext = opt['filename_ext']
        self.task = opt['task']

        keys = []
        total_num_frames = [] # some clips may not have 100 frames
        start_frames = [] # some clips may not start from 00000
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _, start_frame = line.split(' ')
                # match = re.match(r'(.*?)(\d+)$', start_frame)
                # if not match:
                #     raise ValueError(f"Invalid start_frame format: {start_frame}")
                # prefix, start_num_str = match.groups()
                # start_num = int(start_num_str)
                # num_len = len(start_num_str)
                # keys.extend([
                #     f'{folder}/{prefix}{str(start_num + i).zfill(num_len)}'
                #     for i in range(int(start_num), int(start_num)+int(frame_num))
                # ])
                # total_num_frames.extend([int(frame_num)] * int(frame_num))
                # start_frames.extend([start_num] * int(frame_num))
                keys.extend([f'{folder}/{i:{self.filename_tmpl}}' for i in range(int(start_frame), int(start_frame)+int(frame_num))])
                total_num_frames.extend([int(frame_num) for i in range(int(frame_num))])
                start_frames.extend([int(start_frame) for i in range(int(frame_num))])

        val_partition = []

        self.keys = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames = []
        for i, v in zip(range(len(keys)), keys):
            if v.split('/')[0] not in val_partition:
                self.keys.append(keys[i])
                self.total_num_frames.append(total_num_frames[i])
                self.start_frames.append(start_frames[i])

        self.deg_info_dict = {}
        for clip in set(k.split('/')[0] for k in self.keys):
            json_path = self.lq_root / clip /'fog_homo' / self.task / 'degradation_info.json'
            if json_path.exists():
                with open(json_path, 'r') as f:
                    self.deg_info_dict[clip] = json.load(f)
            else:
                self.deg_info_dict[clip] = {}  # 无标签时给空字典

    def __getitem__(self, index):
        
        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames / set interval to fixed 1
        interval = 1

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))


        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        deg_applied_list = []
        deg_beta_list = []
        
        for neighbor in neighbor_list:
            fname = f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
            # img_lq_path = self.lq_root / clip_name / fname
            # img_gt_path = self.gt_root / clip_name / fname
            img_lq_path = self.lq_root / clip_name / 'fog_homo' / self.task / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
            img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get LQ
            img_lq = read_img(img_lq_path)
            img_lqs.append(img_lq)

            # get GT
            img_gt = read_img(img_gt_path)
            img_gts.append(img_gt)

            # 获取退化标签
            json_dict = self.deg_info_dict.get(clip_name, {})
            frame_info = json_dict.get(fname, {})
            deg_applied = frame_info.get('cat', [0]*3)
            deg_fogbeta = frame_info.get('fog_beta', 0)

            deg_applied_list.append(torch.tensor(deg_applied, dtype=torch.uint8))
            deg_beta_list.append(deg_fogbeta)  # params 暂不转 tensor，先保留原格式

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs)

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        deg_applied_tensor = torch.stack(deg_applied_list, dim=0)  # (T, 7)

        return {
            'lq': img_lqs,                # (T, C, H, W)
            'gt': img_gts,                # (T, C, H, W)
            'key': key,
            'deg_applied': deg_applied_tensor,  # (T, 3)
            'deg_params': deg_beta_list         # list of dicts params, 现在由于每个标签参数不一样，所以暂时用不了
        }
    
        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        
        # return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)