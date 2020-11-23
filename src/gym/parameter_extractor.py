from src.common.simple_arg_parse import arg_or_default


def extract_parameters():
    OUTPUT = arg_or_default("--output", default=None)

    comb_lr = arg_or_default("--comb_lr", default=200)
    comb_lower_lr = arg_or_default("--comb_lower_lr", default=0) == 1
    comb_min_proba = arg_or_default("--comb_min_proba", default=0.1)

    twop_lr = arg_or_default("--twop_lr", default=10000)
    twop_lower_lr = arg_or_default("--twop_lower_lr", default=0) == 1
    twop_delta = arg_or_default("--twop_delta", default=0.02)

    offset = arg_or_default("--offset", default=0)

    reward_type = arg_or_default("--reward", default="loss")

    comb_kwargs = {
        'lr': comb_lr,
        'lower_lr': comb_lower_lr,
        'min_proba_thresh': comb_min_proba
    }

    two_point_kwargs = {
        'lr': twop_lr,
        'lower_lr': twop_lower_lr,
        'delta': twop_delta
    }

    return {
        'output': OUTPUT,
        'offset': offset,
        'comb_kwargs': comb_kwargs,
        'two_point_kwargs': two_point_kwargs,
        'reward_type': reward_type
    }
