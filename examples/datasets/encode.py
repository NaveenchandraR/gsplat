import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import imageio.v3 as iio

from seva.modules.autoencoder import AutoEncoder
from seva.utils import load_model

device = "cuda:0"

def encode_images(scene_dir: list):

    # Load AutoEncoder Model
    AE = AutoEncoder(chunk_size=1).to(device)

    # Target Latent resolution
    target_size = (72, 72)

    images = sorted(glob.glob(f"{scene_dir}/images/*"))
    os.makedirs(f"{scene_dir}/latents", exist_ok=True)

    for image_path in tqdm(images, desc="Encoding images"):
        image = iio.imread(image_path)
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0  # [H, W, C]
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Create Latent Image encodigns
        encoded = AE.encode(image_tensor, 1)

        # Resize latent according to the desired shape
        encoded_resized = F.interpolate(encoded, target_size, mode='bilinear', align_corners=False)

        latent_save_path = os.path.splitext(image_path.replace("images", "latents"))[0] + ".pt"
        torch.save(encoded_resized, latent_save_path)

    del AE
    torch.cuda.empty_cache()
