import json
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

import imageio.v3 as iio

import torch
import torch.nn.functional as F


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        all_imgs_path: List[str] = None,
        input_indices: List[str] = None,
        c2ws: List[torch.Tensor] = None,
        Ks: List[torch.Tensor] = None,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize

        self.input_indices = input_indices
        self.c2ws = c2ws
        self.Ks = Ks
        self.image_paths = [img_path.replace("images", "train_images") for img_path in all_imgs_path]  # List[str], (num_images,)
        self.camtoworlds = np.array(c2ws)  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = list(range(len(c2ws)))
        self.image_names = [name.split("/")[-1] for name in self.image_paths]
        self.pt_paths = [
            os.path.splitext(path.replace("train_images", "latents"))[0] + ".pt"
            for path in self.image_paths
        ]

        imsize_dict = dict()  # width, height
        mask_dict = dict()
        Ks_dict = dict()

        for i in self.camera_ids:
            image_path = self.image_paths[i]
            image = iio.imread(image_path)
            imsize_dict[i] = (image.shape[1] // factor, image.shape[0] // factor)
            mask_dict[i] = None
            Ks_dict[i] = Ks[i]

        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K

        # size of the scene measured by cameras
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)



class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        # if split == "train":
        #     self.indices = self.parser.input_indices
        # else:
        #     self.indices = self.parser.camera_ids
        if split == "train":
            self.indices = [i for i in self.parser.input_indices if i % 2 == 0]  # even indices
        else:
            self.indices = [i for i in self.parser.input_indices if i % 2 == 1]  # odd indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        # params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]
        latent_feature = torch.load(self.parser.pt_paths[index], map_location='cpu')
        # latent_feature = latent_feature.squeeze().permute(2, 1, 0)
        latent_feature = latent_feature.squeeze().permute(1, 2, 0)

        # image_tensor = (torch.from_numpy(image).float().
        #                 unsqueeze(0).permute(0, 3, 1, 2) / 255.0)  # [1, C, H, W]

        # resized_image = F.interpolate(image_tensor, size=(72, 72), mode='bilinear', align_corners=False)  # [1, C, 72, 72]
        # resized_image = resized_image.permute(0, 2, 3, 1).squeeze(0) # Back to [1, 72, 72, C]
        resized_image = (torch.from_numpy(image).float())

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": resized_image,
            "image_id": item,  # the index of the image in the dataset
            "latent_feature": latent_feature,
        }
        # import ipdb; # ipdb.set_trace()
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
