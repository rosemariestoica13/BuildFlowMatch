import logging

# thirdparty/DepthAnythingV2's dinov2_layers log a "xFormers not available"
# warning at import time; xformers is an optional speed-up we don't install.
logging.getLogger("dinov2").setLevel(logging.ERROR)

from model.build_match_flow import BuildMatchFlow


def fetch_model(args):
    if args.algorithm == 'waft-a2':
        model = BuildMatchFlow(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model
