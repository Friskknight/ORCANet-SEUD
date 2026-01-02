import glob
import torch
import json
from os import path as osp
from torch.utils import data as data

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq, read_img_seq_iqa
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from pyiqa.utils.img_util import imread2tensor


@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    ::

        dataroot
        ├── subfolder1
            ├── frame000
            ├── frame001
            ├── ...
        ├── subfolder2
            ├── frame000
            ├── frame001
            ├── ...
        ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwarg.
        cache_data (bool): Whether to cache testing datasets.
        name (str): Dataset name.
        meta_info_file (str): The path to the file storing the list of test folders. If not provided, all the folders
            in the dataroot will be used.
        num_frame (int): Window size for input frames.
        padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_dp, self.imgs_gt, self.jsonfile = {}, {}, {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            # img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
            # img_paths_lq = sorted(glob.glob(osp.join(subfolder_lq, 'fog_homo', opt['task'], '*.jpg')))
            img_paths_lq = sorted(
                glob.glob(osp.join(subfolder_lq, 'fog_homo', opt['task'], '*.jpg'))
                + glob.glob(osp.join(subfolder_lq, 'fog_homo', opt['task'], '*.png'))
            )
            img_paths_dp = sorted(glob.glob(osp.join(subfolder_lq, 'depth_pred', '*.png')))
            img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))
            img_paths_js = osp.join(self.lq_root, subfolder_name, 'fog_homo', opt['task'], 'degradation_info.json')

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                    f' and gt folders ({len(img_paths_gt)})')
            # assert max_idx == len(img_paths_dp), (f'Different number of images in lq ({max_idx})'
            #                                         f' and depth folders ({len(img_paths_dp)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            # self.data_info['dp_path'].extend(img_paths_dp)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                # self.imgs_dp[subfolder_name] = read_img_seq(img_paths_dp)
                self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                self.jsonfile[subfolder_name] = img_paths_js
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                # self.imgs_dp[subfolder_name] = img_paths_dp
                self.imgs_gt[subfolder_name] = img_paths_gt
                self.jsonfile[subfolder_name] = img_paths_js


    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])

@DATASET_REGISTRY.register()
class VideoRecurrentTestDataset(VideoTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        opt (dict): Same as VideoTestDataset. Unused opt:
        padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
        self.folders = sorted(list(set(self.data_info['folder'])))
        self.chunk_size = 80
        self.key_frame = 6

        # 预切所有 folder -> slices: [{'folder': f, 'start': s, 'end': e}, ...]
        self.slices = []
        for f in self.folders:
            flen = self._get_folder_len(f)
            if flen <= 0:
                continue
            if flen <= self.chunk_size:
                self.slices.append({'folder': f, 'start': 0, 'end': flen})
            else:
                s = 0
                while s < flen:
                    if s + self.chunk_size + self.chunk_size//2 >= flen:
                        e = flen
                    else:
                        e = s + self.chunk_size
                    self.slices.append({'folder': f, 'start': s, 'end': e})
                    s = e

    def _get_folder_len(self, folder):
        if self.cache_data:
            return int(self.imgs_lq[folder].shape[0])
        else:
            return int(len(self.imgs_lq[folder]))

    def __getitem__(self, index):
        # 现在 index 指向的是一个“切片”而非“整个folder”
        sl = self.slices[index]
        folder = sl['folder']
        s, e = sl['start'], sl['end']
        json_path = self.jsonfile[folder]


        if self.cache_data:
            # 直接在缓存Tensor上切片（按时间维度 T）
            imgs_lq = self.imgs_lq[folder][s:e]
            # imgs_dp = self.imgs_dp[folder][s:e]
            imgs_gt = self.imgs_gt[folder][s:e]
        else:
            # 只读该子段的路径列表；read_img_seq 应该能接收“子列表”
            imgs_lq = read_img_seq(self.imgs_lq[folder][s:e])
            imgs_gt = read_img_seq(self.imgs_gt[folder][s:e])
            imgs_gt.squeeze_(0)
        
        if osp.exists(json_path):
            with open(json_path, 'r') as f:
                deg_info_dict = json.load(f)
            sorted_keys = sorted(deg_info_dict.keys())
            cats_merged = [deg_info_dict[k]["cat"] for k in sorted_keys]
            cats_merged = cats_merged[s:e]
            cats_sampled = cats_merged[::self.key_frame]
            if cats_sampled[-1] is not cats_merged[-1]:
                cats_sampled.append(cats_merged[-1])
            return {
                'lq': imgs_lq,
                'gt': imgs_gt,
                'degapp': cats_sampled,
                'folder': folder,
                'start_idx': s,
            }
        else:
            return {
                'lq': imgs_lq,
                'gt': imgs_gt,
                'folder': folder,
                'start_idx': s,
            }


    def __folderlen__(self, index):
        return self._get_folder_len(self.folders[index])

    def __len__(self):
        return len(self.slices)
    
# @DATASET_REGISTRY.register()
# class VideoRecurrentTestDataset(VideoTestDataset):
#     """Video test dataset for recurrent architectures, which takes LR video
#     frames as input and output corresponding HR video frames.

#     Args:
#         opt (dict): Same as VideoTestDataset. Unused opt:
#         padding (str): Padding mode.

#     """

#     def __init__(self, opt):
#         super(VideoRecurrentTestDataset, self).__init__(opt)
#         # Find unique folder strings
#         self.folders = sorted(list(set(self.data_info['folder'])))

#     def __getitem__(self, index, selectindex=None):
#         folder = self.folders[index]
#         # folder = self.data_info['folder'][index]

#         if self.cache_data:
#             imgs_lq = self.imgs_lq[folder]
#             imgs_dp = self.imgs_dp[folder]
#             imgs_gt = self.imgs_gt[folder]
#         else:
#             imgs_lq = read_img_seq(self.imgs_lq[folder])
#             imgs_dp = read_img_seq(self.imgs_dp[folder])
#             imgs_gt = read_img_seq(self.imgs_gt[folder])
#             imgs_gt.squeeze_(0)

#             # raise NotImplementedError('Without cache_data is not implemented.')

#         return {
#             'lq': imgs_lq,
#             'dp': imgs_dp,
#             'gt': imgs_gt,
#             'folder': folder,
#         }
    
#     def __folderlen__(self, index):
#         return len(self.imgs_lq[self.folders[index]])
    
#     def __len__(self):
#         return len(self.folders)
