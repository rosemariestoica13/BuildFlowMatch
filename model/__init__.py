from model.build_match_flow import BuildMatchFlow


def fetch_model(args):
    if args.algorithm == 'waft-a2':
        model = BuildMatchFlow(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model
