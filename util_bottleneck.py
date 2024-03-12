from interaction.bottleneck import *

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