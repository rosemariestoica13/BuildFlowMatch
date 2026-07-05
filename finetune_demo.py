"""Minimal fine-tuning demo for WAFT (waft-a2 / dav2 backbone).

Loads a pretrained checkpoint, fine-tunes it for a handful of steps on a
small custom dataset (image1/image2/flow folders, see dataloader/custom.py),
then runs inference before vs. after fine-tuning and saves flow
visualizations so you can see the difference.

Example (run from inside Colab, see colab_finetune_demo.ipynb):

    python finetune_demo.py \
        --cfg config/a2/dav2/sintel-gm.json \
        --ckpt checkpoints/sintel-gm-final.pth \
        --data_dir data/custom \
        --steps 100 \
        --out_ckpt checkpoints/finetuned.pth
"""
import argparse
import itertools
import os

import cv2
import torch
import torch.optim as optim
import torch.utils.data as data

from config.parser import parse_args
from model import fetch_model
from dataloader.custom import CustomFlowDataset
from criterion.loss import sequence_loss
from utils.utils import load_ckpt
from utils.flow_viz import flow_to_image
from inference_tools import InferenceWrapper


def build_loader(args, data_dir):
    aug_params = {
        'crop_size': args.image_size,
        'min_scale': -0.1,
        'max_scale': 0.3,
        'do_flip': False,
    }
    dataset = CustomFlowDataset(aug_params, root=data_dir)
    print(f"Custom dataset: {len(dataset)} image pair(s) found in {data_dir}")
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    return dataset, loader


@torch.no_grad()
def save_flow_preview(model, dataset, out_path, device):
    image1, image2, flow_gt, valid = dataset[0]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    output = model.calc_flow(image1, image2)
    flow = output['flow'][-1]
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(out_path, flow_vis)
    epe = ((flow[0] - flow_gt.to(device)) ** 2).sum(0).sqrt()
    epe = (epe * valid.to(device)).sum() / valid.to(device).sum().clamp(min=1)
    print(f"[{out_path}] EPE vs ground truth: {epe.item():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='model config json (e.g. config/a2/dav2/sintel-gm.json)')
    parser.add_argument('--ckpt', required=True, type=str, help='pretrained checkpoint to start from')
    parser.add_argument('--data_dir', required=True, type=str, help='folder with image1/ image2/ flow/ subfolders')
    parser.add_argument('--steps', default=100, type=int, help='number of fine-tuning steps')
    parser.add_argument('--lr', default=None, type=float, help='override learning rate from config')
    parser.add_argument('--out_ckpt', default='checkpoints/finetuned.pth', type=str)
    parser.add_argument('--preview_dir', default='demo_out', type=str)
    args = parse_args(parser)
    if args.lr:
        args.lr = args.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.preview_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_ckpt) or '.', exist_ok=True)

    model = fetch_model(args).to(device)
    load_ckpt(model, args.ckpt)
    print(f"Loaded checkpoint from {args.ckpt}")

    dataset, loader = build_loader(args, args.data_dir)

    # --- preview before fine-tuning ---
    model.eval()
    wrapped = InferenceWrapper(model)
    save_flow_preview(wrapped, dataset, os.path.join(args.preview_dir, 'flow_before.jpg'), device)

    # --- fine-tune ---
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    step = 0
    for image1, image2, flow, valid in itertools.cycle(loader):
        if step >= args.steps:
            break
        image1, image2, flow, valid = image1.to(device), image2.to(device), flow.to(device), valid.to(device)

        optimizer.zero_grad()
        output = model(image1, image2, flow_gt=flow)
        loss = sequence_loss(output, flow, valid, args.gamma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if valid.sum() > 0:
            epe = (((flow - output['flow'][-1]) ** 2).sum(dim=1)).sqrt()
            epe = (epe * valid).sum() / valid.sum()
            print(f"step {step:04d} | loss {loss.item():.4f} | epe {epe.item():.3f}")
        else:
            print(f"step {step:04d} | loss {loss.item():.4f}")
        step += 1

    torch.save(model.state_dict(), args.out_ckpt)
    print(f"Saved fine-tuned checkpoint to {args.out_ckpt}")

    # --- preview after fine-tuning ---
    model.eval()
    wrapped = InferenceWrapper(model)
    save_flow_preview(wrapped, dataset, os.path.join(args.preview_dir, 'flow_after.jpg'), device)


if __name__ == '__main__':
    main()
