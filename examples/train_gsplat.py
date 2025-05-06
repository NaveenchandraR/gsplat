import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml

import sys

sys.path.append("/data/naveen_ankit_phd/personal/gsplat/examples")

from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

import sys

# sys.path.append("/data/naveen_ankit_phd/nerfstudio/gsplat")
from gsplat.exporter import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.utils import save_ply
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

# Import classes
from simple_trainer import Runner, Config
# import functions
from simple_trainer import create_splats_with_optimizers

import ipdb


def opengl_to_colmap_extrinsics(c2ws: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Converts a list of [3, 4] OpenGL camera-to-world extrinsic matrices
    to COLMAP coordinate system format.

    Args:
        c2ws (List[torch.Tensor]): List of [3, 4] OpenGL extrinsic matrices.

    Returns:
        List[torch.Tensor]: Converted [3, 4] COLMAP extrinsic matrices.
    """
    assert all(c2w.shape == (3, 4) for c2w in c2ws), "Each matrix must be [3, 4] shape."

    T = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=c2ws[0].dtype, device=c2ws[0].device))

    c2ws_colmap = []
    for c2w in c2ws:
        R = c2w[:, :3]
        t = c2w[:, 3]
        R_colmap = T @ R
        t_colmap = T @ t
        c2w_colmap = torch.cat([R_colmap, t_colmap.unsqueeze(1)], dim=1)
        bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=c2w.dtype, device=c2w.device)
        c2w_colmap = torch.cat([c2w_colmap, bottom_row], dim=0)  # [4, 4]
        c2ws_colmap.append(c2w_colmap)

    return c2ws_colmap


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)
    # ipdb.set_trace()

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    runner.viewer.complete()
    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(10)


def train_gaussians_from_input(
        all_imgs_path,
        input_indices,
        c2ws,
        Ks,
        config_name: Literal["default", "mcmc"] = "default",
        config_overrides: Optional[dict] = None,
):
    # ipdb.set_trace()

    # Convert OpenGL to Colmap
    c2ws = opengl_to_colmap_extrinsics(c2ws)
    # ipdb.set_trace()

    # # Config objects we can choose between.
    # # Each is a tuple of (CLI description, config object).
    # configs = {
    #     "default": (
    #         "Gaussian splatting training using densification heuristics from the original paper.",
    #         Config(
    #             strategy=DefaultStrategy(verbose=True),
    #         ),
    #     ),
    #     "mcmc": (
    #         "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
    #         Config(
    #             init_opa=0.5,
    #             init_scale=0.1,
    #             opacity_reg=0.01,
    #             scale_reg=0.01,
    #             strategy=MCMCStrategy(verbose=True),
    #         ),
    #     ),
    # }
    cfg = Config(
                strategy=DefaultStrategy(verbose=True),
            )
    # ipdb.set_trace()

    cfg.all_imgs_path = all_imgs_path
    cfg.input_indices = input_indices
    cfg.c2ws = np.array([t.cpu().numpy() for t in c2ws])
    cfg.Ks = np.array([t.cpu().numpy() for t in Ks])

    # Apply overrides, if any
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)

    cfg.adjust_steps(cfg.steps_scaler)
    # ipdb.set_trace()

    cli(main, cfg, verbose=True)

