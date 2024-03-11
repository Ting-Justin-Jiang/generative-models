import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import time
from util_debug import *
import os
import argparse


MAX_BS = 4 * 100


def generate_order_interaction_img(args, model: torch.nn.Module, feature: torch.Tensor, feature_shape,
                                  name: str, pairs: np.ndarray,
                                  ratio: float, player_io_handler: PlayerIoHandler,
                                  interaction_logit_io_handler: InteractionLogitIoHandler):
    """
    Input:
        args: args
        model: nn.Module, model to be evaluated
        feature: (1,C,H,W) tensor
        feature_shape: tuple, shape of feature
        name: str, name of this sample
        pairs: (pairs_num, 2) array, (i,j) pairs
        ratio: float, ratio of the order of the interaction, order=(n-2)*ratio
        player_io_handler:
        interaction_logit_io_handler:
    Return:
        None
    """
    time0 = time.time()
    model.to(args.device)
    order = int((args.grid_size ** 2 - 2) * ratio)
    print("m=%d" % order)

    with torch.no_grad():
        model.eval()
        channels = feature.size(1)
        players = player_io_handler.load(round(ratio * 100), name)
        ori_logits = []

        forward_mask = []
        for index, pair in enumerate(pairs):
            print('\r\t\tPairs: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(pairs)), end='')
            point1, point2 = pair[0], pair[1]

            players_curr_pair = players[index] # context S for this pair of (i,j)
            mask = torch.zeros((4 * args.samples_number_of_s, channels, args.grid_size ** 2), device=args.device)

            if order != 0: # if order == 0, then S=emptyset, we don't need to set S
                S_cardinality = players_curr_pair.shape[1]  # |S|
                assert S_cardinality == order
                idx_multiple_of_4 = 4 * np.arange(args.samples_number_of_s)  # indices: 0, 4, 8...
                stack_idx = np.stack([idx_multiple_of_4] * S_cardinality, axis=1)  # stack the indices to match the shape of player_curr_i
                mask[stack_idx, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+1, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+2, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+3, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)

            mask[4 * np.arange(args.samples_number_of_s) + 1, :, point1] = 1  # S U {i}
            mask[4 * np.arange(args.samples_number_of_s) + 2, :, point2] = 1  # S U {j}
            mask[4 * np.arange(args.samples_number_of_s), :, point1] = 1  # S U {i,j}
            mask[4 * np.arange(args.samples_number_of_s), :, point2] = 1  # S U {i,j}

            mask = mask.view(4 * args.samples_number_of_s, channels, args.grid_size, args.grid_size)
            mask = F.interpolate(mask.clone(), size=[feature_shape[2], feature_shape[3]], mode='nearest').float()


            if len(mask) > MAX_BS: # if sample number of S is too large (especially for vgg19), we need to split one batch into several iterations
                iterations = math.ceil(len(mask) / MAX_BS)
                for it in range(iterations): # in each iteration, we compute output for MAX_BS images
                    batch_mask = mask[it * MAX_BS : min((it+1) * MAX_BS, len(mask))]
                    expand_feature = feature.expand(len(batch_mask), channels, feature_shape[2], feature_shape[3]).clone()
                    masked_feature = batch_mask * expand_feature

                    output_ori = model(masked_feature)
                    assert not torch.isnan(output_ori).any(), 'there are some nan numbers in the model output'
                    ori_logits.append(output_ori.detach())

            else: # if sample number of S is small, we can concatenate several batches and do a single inference
                forward_mask.append(mask)
                if (len(forward_mask) < args.cal_batch // args.samples_number_of_s) and (index < args.pairs_number - 1):
                    continue
                else:
                    forward_batch = len(forward_mask) * args.samples_number_of_s
                    batch_mask = torch.cat(forward_mask, dim=0)
                    expand_feature = feature.expand(4 * forward_batch, channels, feature_shape[2], feature_shape[3]).clone()
                    masked_feature = batch_mask * expand_feature

                    output_ori = model(masked_feature)
                    assert not torch.isnan(output_ori).any(), 'there are some nan numbers in the model output'

                    ori_logits.append(output_ori.detach())
                    forward_mask = []
        print('done time: ', time.time() - time0)

        all_logits = torch.cat(ori_logits, dim=0)  # (pairs_num*4*samples_number_of_s, class_num)
        print("all_logits shape: ", all_logits.shape)
        interaction_logit_io_handler.save(round(ratio * 100), name, all_logits)


def generate_interactions(args, model: nn.Module, dataloader: DataLoader, pair_io_handler: PairIoHandler,
                         player_io_handler: PlayerIoHandler, interaction_logit_io_handler: InteractionLogitIoHandler):
    model.to(args.device)

    with torch.no_grad():
        model.eval()
        total_pairs = pair_io_handler.load()
        for index, (name, image, label) in enumerate(dataloader):
            print('Images: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(dataloader)))

            image = image.to(args.device)
            label = label.to(args.device)

            pairs = total_pairs[index]

            for ratio in args.ratios:
                print('\tCurrent ratio: \033[1;31m\033[5m%.2f' % ratio)
                order = int((args.grid_size ** 2 - 2) * ratio)
                seed_torch(1000 * index + order + args.seed)
                compute_order_interaction_img(args, model, image, image.shape, name[0], pairs, ratio, player_io_handler,
                                              interaction_logit_io_handler)


def compute_order_interaction_img(args, name: str, label: torch.Tensor, ratio: float,
                                  interaction_logit_io_handler: InteractionLogitIoHandler,
                                  interaction_io_handler: InteractionIoHandler):
    """
    Input:
        args: args
        name: str, name of this sample
        label: (1,) tensor, label of this sample
        ratio: float, ratio of the order of the interaction, order=(n-2)*ratio
        interaction_logit_io_handler:
        interaction_io_handler:
    Return:
        None
    """
    interactions = []

    logits = interaction_logit_io_handler.load(round(ratio * 100), name)
    logits = logits.reshape((args.pairs_number, args.samples_number_of_s * 4, args.class_number)) # load saved logits

    for index in range(args.pairs_number):
        print('\r\t\tPairs: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, args.pairs_number), end='')
        output_ori = logits[index, :, :]

        v = get_reward(args, output_ori, label)  # (4*samples_number_of_s,)

        # Delta v(i,j,S) = v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)
        score_ori = v[4 * np.arange(args.samples_number_of_s)] + v[4 * np.arange(args.samples_number_of_s) + 3] \
                    - v[4 * np.arange(args.samples_number_of_s) + 1] - v[4 * np.arange(args.samples_number_of_s) + 2]
        interactions.extend(score_ori.tolist())

    print('')
    interactions = np.array(interactions).reshape(-1, args.samples_number_of_s) # (pair_num, sample_num)
    assert interactions.shape[0] == args.pairs_number

    interaction_io_handler.save(round(ratio * 100), name, interactions)  # (pair_num, sample_num)


def compute_interactions(args, model: nn.Module, dataloader: DataLoader, interaction_logit_io_handler: InteractionLogitIoHandler, interaction_io_handler: InteractionIoHandler):
    model.to(args.device)

    with torch.no_grad():
        model.eval()
        for index, (name, image, label) in enumerate(dataloader):
            print('Images: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(dataloader)))
            image = image.to(args.device)
            label = label.to(args.device)

            for ratio in args.ratios:
                print('\tCurrent ratio: \033[1;31m\033[5m%.2f' % ratio)
                order = int((args.grid_size ** 2 - 2) * ratio)
                seed_torch(1000 * index + order + args.seed)
                if args.out_type == 'gt':
                    compute_order_interaction_img(args, name[0], label, ratio, interaction_logit_io_handler, interaction_io_handler)
                else:
                    raise Exception(f"output type [{args.out_type}] not supported.")


def main():
    ...