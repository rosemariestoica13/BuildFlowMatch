"""Standalone inference demo for BuildFlowMatch (waft-a2 / dav2 backbone).

Runs optical flow inference on a single image pair (no ground truth needed)
and saves a flow visualization.

Example:
    python infer_demo.py \
        --cfg config/a2/dav2/sintel-gm.json \
        --ckpt checkpoints/finetuned.pth \
        --img1 infer_data/000008_10.png \
        --img2 infer_data/000008_11.png \
        --out demo_out/infer_flow.jpg
"""
import argparse
import os

import cv2
import torch

from config.parser import parse_args
from model import fetch_model
from utils.utils import load_ckpt
from utils.flow_viz import flow_to_image
from inference_tools import InferenceWrapper


def load_image(path, device):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='model config json (e.g. config/a2/dav2/sintel-gm.json)')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoint to run inference with')
    parser.add_argument('--img1', required=True, type=str)
    parser.add_argument('--img2', required=True, type=str)
    parser.add_argument('--out', default='demo_out/infer_flow.jpg', type=str)
    args = parse_args(parser)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    model = fetch_model(args).to(device)
    load_ckpt(model, args.ckpt)
    print(f"Loaded checkpoint from {args.ckpt}")
    model.eval()
    wrapped = InferenceWrapper(model)

    image1 = load_image(args.img1, device)
    image2 = load_image(args.img2, device)

    with torch.no_grad():
        output = wrapped.calc_flow(image1, image2)
    flow = output['flow'][-1]
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(args.out, flow_vis)
    print(f"Saved flow visualization to {args.out}")


if __name__ == '__main__':
    main()
