import os.path as osp
from glob import glob

import numpy as np

from utils import frame_utils
from dataloader.template import FlowDataset


class CustomFlowDataset(FlowDataset):
    """Generic flow dataset for user-provided image pairs + ground-truth flow.

    Expected folder layout under `root`:

        root/
          image1/000000.png   000001.png   ...
          image2/000000.png   000001.png   ...
          flow/000000.flo     000001.flo   ...

    - image1[i] and image2[i] form a pair; flow[i] is the ground-truth flow
      from image1[i] to image2[i].
    - Files are matched by sorted order, so keep filenames aligned across the
      three folders (same basenames recommended, extensions can differ).
    - Ground truth is either a `.flo` file (Middlebury format, see
      utils/frame_utils.py:writeFlow / readFlow) or a `.png` file in KITTI's
      16-bit encoding (see utils/frame_utils.py:readFlowKITTI), e.g. the
      flow_occ/flow_noc folders from the KITTI scene flow devkit.
    """

    def __init__(self, aug_params=None, root='data/custom'):
        super().__init__(aug_params)
        images1 = sorted(glob(osp.join(root, 'image1', '*')))
        images2 = sorted(glob(osp.join(root, 'image2', '*')))
        flows = sorted(glob(osp.join(root, 'flow', '*')))

        if len(images1) == 0:
            raise FileNotFoundError(f"No images found in {osp.join(root, 'image1')}")
        if not (len(images1) == len(images2) == len(flows)):
            raise ValueError(
                f"image1/image2/flow must have the same number of files, "
                f"got {len(images1)}/{len(images2)}/{len(flows)}"
            )

        for im1, im2 in zip(images1, images2):
            self.image_list += [[im1, im2]]
        self.flow_list += flows

    def read_flow(self, index):
        path = self.flow_list[index]
        if path.endswith('.png'):
            return frame_utils.readFlowKITTI(path)
        flow = frame_utils.read_gen(path)
        valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
        return flow, valid
