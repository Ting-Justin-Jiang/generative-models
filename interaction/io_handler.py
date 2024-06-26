import os
from typing import Any, Iterable, Tuple

import numpy as np
import torch
from interaction.interaction_common import config


def set_args(args):
    args.prefix = config['prefix']
    args.ratios = config['ratios']
    args.stride = config['stride']  # radius of neighborhood for sampling (i,j) pair
    args.datasets_dirname = config['datasets_dirname']
    args.pretrained_models_dirname = config['pretrained_models_dirname']

    # ---- set sample number of ij, S and images ------
    # compute interaction between image pixels/patches (including all channels)
    if args.inter_type == "pixel":
        args.pairs_number = config['pairs_number_pixel']
        args.samples_number_of_s = config['samples_number_of_s_pixel']
        args.selected_img_number = config['selected_img_number_pixel']
        args.output_dirname = args.output_dirname + "_INTER_pixel_CLASS_%s_GRID_%dx%d" % (
        args.chosen_class, args.grid_size, args.grid_size)
    else:
        raise Exception("Not a valid output_dirname")

    args.output_dirname = args.output_dirname + "_seed%d" % args.seed

    # ======= model checkpoint path ======
    if args.arch == "our_alexnet_cifar10_normal_lr0.01_log1_da_flip_crop_best":
        args.checkpoint_path = os.path.join(args.prefix,
                                            "checkpoints/our_alexnet_cifar10/normal_lr0.01_log1_da_flip_crop_seed0/model_best.pth")

    # L+ and L- models
    elif args.arch == "our_alexnet_cifar10_dp_pos0_entropy_deltav_baseline_0.5_0.0_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best":
        args.checkpoint_path = os.path.join(args.prefix,
                                            "checkpoints/our_alexnet_cifar10/dp_pos0_entropy_deltav_baseline_0.5_0.0_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_seed0/model_best.pth")
    elif args.arch == "our_alexnet_cifar10_dp_pos0_deltav_baseline_0.7_0.3_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best":
        args.checkpoint_path = os.path.join(args.prefix,
                                            "checkpoints/our_alexnet_cifar10/dp_pos0_deltav_baseline_0.7_0.3_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_seed0/model_best.pth")
    elif args.arch == "our_alexnet_cifar10_dp_pos0_entropy_deltav_baseline_1.0_0.7_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best":
        args.checkpoint_path = os.path.join(args.prefix,
                                            "checkpoints/our_alexnet_cifar10/dp_pos0_entropy_deltav_baseline_1.0_0.7_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_seed0/model_best.pth")

    else:
        raise Exception(f"Model [{args.arch}] not implemented. Error in set_args.")

    # ------ dataset setting -----
    if args.dataset == "cifar10":
        args.class_number = 10
        args.image_size = 32
    else:
        raise Exception(f"dataset [{args.dataset}] not implemented. Error in set_args.")

    # ----- save paths -----
    args.figures_dirname = config['figures_dirname']
    args.output_dir = os.path.join(args.prefix, "results", args.output_dirname,
                                   "MODEL_%s_DATA_%s" % (args.arch, args.dataset))

    args.samples_dir = os.path.join(args.output_dir, config['samples_dirname'])
    args.samples_file = os.path.join(args.samples_dir, config['samples_filename'])

    args.pairs_dir = os.path.join(args.output_dir, config['pairs_dirname'])
    args.pairs_file = os.path.join(args.pairs_dir, config['pairs_filename'])
    args.players_dir = os.path.join(args.pairs_dir, config['players_dirname'])
    args.players_dir_with_ratio_pattern = os.path.join(args.players_dir, 'ratio%d')
    args.players_file_pattern = os.path.join(args.players_dir_with_ratio_pattern, '%s.npy')

    args.interactions_logit_dir = os.path.join(args.output_dir, config['interactions_logit_dirname'])
    args.interactions_logit_dir_with_ratio_pattern = os.path.join(args.interactions_logit_dir, 'ratio%d')
    args.interactions_logit_file_pattern = os.path.join(args.interactions_logit_dir_with_ratio_pattern, '%s.pth')

    if hasattr(args, 'softmax_type') and hasattr(args, 'out_type'):
        args.interactions_dir = os.path.join(args.output_dir, config['interactions_dirname'] + "_out_%s_softmax_%s" % (
            args.out_type, args.softmax_type))
        args.interactions_dir_with_ratio_pattern = os.path.join(args.interactions_dir, 'ratio%d')
        args.interactions_file_pattern = os.path.join(args.interactions_dir_with_ratio_pattern, '%s.npy')


class BaseIoHandler:

    def __init__(self, root: str) -> None:
        self.root = root
        if not os.path.isdir(root):
            os.makedirs(root)

    def save(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def load(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class SampleIoHandler(BaseIoHandler):

    def __init__(self, args):
        super().__init__(args.samples_dir)
        self.file = args.samples_file
        self.dataset = args.dataset

    def save(self, data):
        with open(self.file, 'w', encoding='UTF-8') as f:
            # CIFAR10 format: class_name, img index(in the WHOLE dataset), class_index
            f.write('\n'.join(map(lambda item: f'{item[0]},{item[1]},{item[2]}', data)))

    def load(self):
        data = []
        with open(self.file, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                item = line.strip().split(',')

                if self.dataset == "cifar10":
                    # CIFAR10 format: class_name, img index(in the WHOLE dataset, 0-based), class_index
                    data.append((item[0], int(item[1]), int(item[2])))
                else:
                    raise Exception(f"dataset [{self.dataset}] not implemented. Error in SampleIoHandler.")

        return data


class PairIoHandler(BaseIoHandler):  # load and save a player pair (i,j)

    def __init__(self, args) -> None:
        super().__init__(args.pairs_dir)
        self.file = args.pairs_file

    def save(self, data: np.ndarray) -> None:
        np.save(self.file, data)

    def load(self) -> np.ndarray:
        return np.load(self.file)


class PlayerIoHandler(BaseIoHandler):  # load and save a context S

    def __init__(self, args) -> None:
        super().__init__(args.players_dir)
        self.players_dir_with_ratio_pattern = args.players_dir_with_ratio_pattern
        self.players_file_pattern = args.players_file_pattern

    def save(self, ratio, name: str, data: np.ndarray) -> None:
        players_dir_with_ratio = self.players_dir_with_ratio_pattern % ratio
        if not os.path.isdir(players_dir_with_ratio):
            os.makedirs(players_dir_with_ratio)
        players_file = self.players_file_pattern % (ratio, name)
        np.save(players_file, data)

    def load(self, ratio, name: str) -> np.ndarray:
        players_file = self.players_file_pattern % (ratio, name)
        return np.load(players_file)


class InteractionLogitIoHandler(BaseIoHandler):  # load and save logits

    def __init__(self, args) -> None:
        super().__init__(args.interactions_logit_dir)
        self.interactions_logit_dir_with_ratio_pattern = args.interactions_logit_dir_with_ratio_pattern
        self.interactions_logit_file_pattern = args.interactions_logit_file_pattern
        self.device = args.device

    def save(self, ratio, name: str, data: torch.Tensor) -> None:
        interaction_logit_dir_with_ratio = self.interactions_logit_dir_with_ratio_pattern % ratio
        if not os.path.isdir(interaction_logit_dir_with_ratio):
            os.makedirs(interaction_logit_dir_with_ratio)
        interaction_logit_file = self.interactions_logit_file_pattern % (ratio, name)
        torch.save(data, interaction_logit_file)

    def load(self, ratio, name: str) -> torch.Tensor:
        interaction_logit_file = self.interactions_logit_file_pattern % (ratio, name)
        return torch.load(interaction_logit_file, map_location=self.device)


class InteractionIoHandler(BaseIoHandler):  # load and save interactions

    def __init__(self, args) -> None:
        super().__init__(args.interactions_dir)
        self.interactions_dir_with_ratio_pattern = args.interactions_dir_with_ratio_pattern
        self.interactions_file_pattern = args.interactions_file_pattern

    def save(self, ratio, name: str, data: np.ndarray) -> None:
        interaction_dir_with_ratio = self.interactions_dir_with_ratio_pattern % ratio
        if not os.path.isdir(interaction_dir_with_ratio):
            os.makedirs(interaction_dir_with_ratio)
        interaction_file = self.interactions_file_pattern % (ratio, name)
        np.save(interaction_file, data)

    def load(self, ratio, name: str) -> np.ndarray:
        interaction_file = self.interactions_file_pattern % (ratio, name)
        return np.load(interaction_file)