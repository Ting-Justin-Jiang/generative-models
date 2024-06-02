from interaction.bottleneck import *


def get_reward(args, logits, label):
    """ given logits, calculate reward score for interaction computation
    Input:
        args: args.softmax_type determines which type of score to compute the interaction
            - normal: use log p, p is the probability of the {label} class
            - modified: use log p/(1-p), p is the probability of the {label} class
            - yi: use logits the {label} class
        logits: (N,num_class) tensor, a batch of logits before the softmax layer
        label: (1,) tensor, ground truth label
    Return:
        v: (N,) tensor, reward score
    """
    if args.softmax_type == "normal": # log p
        v = F.log_softmax(logits, dim=1)[:, label[0]]
    elif args.softmax_type == "modified": # log p/(1-p)
        v = logits[:, label[0]] - torch.logsumexp(logits[:, np.arange(args.class_number) != label[0].item()],dim=1)
    elif args.softmax_type == "yi": # logits
        v = logits[:, label[0]]
    else:
        raise Exception(f"softmax type [{args.softmax_type}] not implemented")
    return v


def gen_pairs(grid_size: int, pair_num: int, stride: int = 1) -> np.ndarray:
    """
    Input:
        grid_size: int, the image is partitioned to grid_size * grid_size patches. Each patch is considered as a player.
        pair_num: int, how many (i,j) pairs to sample for one image
        stride: int, j should be sampled in a neighborhood of i. stride is the radius of the neighborhood.
            e.g. if stride=1, then j should be sampled from the 8 neighbors around i
                if stride=2, then j should be sampled from the 24 neighbors around i
    Return:
        total_pairs: (pair_num,2) array, sampled (i,j) pairs
    """

    neighbors = [(i, j) for i in range(-stride, stride + 1)
                 for j in range(-stride, stride + 1)
                 if
                 i != 0 or j != 0]

    total_pairs = []
    for _ in range(pair_num):
        while True:
            x1 = np.random.randint(0, grid_size)
            y1 = np.random.randint(0, grid_size)
            point1 = x1 * grid_size + y1

            neighbor = random.choice(neighbors)
            x2 = clamp(x1 + neighbor[0], 0, grid_size - 1)
            y2 = clamp(y1 + neighbor[1], 0, grid_size - 1)
            point2 = x2 * grid_size + y2

            if point1 == point2:
                continue

            if [point1, point2] in total_pairs or [point2, point1] in total_pairs:
                continue
            else:
                total_pairs.append(list([point1, point2]))
                break

    return np.array(total_pairs)


# generate pairs information for all prompts in the sampling trial
def call_gen_pairs(args, prompts):
    # TODO make sure the pairs generate MUST be kept unmerged (belong to src partition) in TOME
    # TODO grid_size corresponds to in one token for TOME
    total_pairs = []

    pair_io_handler = PairIoHandler(args)
    player_io_handler = PlayerIoHandler(args)

    for index, prompt in enumerate(prompts):
        print('\rPairs: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(prompts)), end='')

        seed_torch(1000 * index + args.seed)  # seed for sampling (i,j) pair
        pairs = gen_pairs(args.grid_size, args.pairs_number, args.stride)
        for ratio in args.ratios:
            m = int((args.grid_size ** 2 - 2) * ratio)  # m-order

            seed_torch(1000 * index + m + 1 + args.seed)  # seed for sampling context S
            players_with_ratio = []
            for pair in pairs:
                point1, point2 = pair[0], pair[1]
                context = list(range(args.grid_size ** 2))
                context.remove(point1)
                context.remove(point2)

                curr_players = []
                for _ in range(args.samples_number_of_s):
                    curr_players.append(np.random.choice(context, m, replace=False))  # sample contexts of cardinality m

                players_with_ratio.append(curr_players)
            players_with_ratio = np.array(players_with_ratio)  # (pair_num, sample_num_of_s, m), contexts S of cardinality m for different (i,j) pairs
            print(players_with_ratio.shape)
            player_io_handler.save(round(ratio * 100), prompt, players_with_ratio)
        total_pairs.append(pairs)

    total_pairs = np.array(total_pairs)  # (num_imgs, num_pairs, 2), all (i,j) pairs
    print(total_pairs.shape)
    pair_io_handler.save(total_pairs)
    print('\nDone!')