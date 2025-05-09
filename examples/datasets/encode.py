import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import imageio.v3 as iio
from PIL import Image

from seva.modules.autoencoder import AutoEncoder
from seva.utils import load_model

import ipdb

device = "cuda:0"

def encode_images(scene_dir: str):

    # Load AutoEncoder Model
    AE = AutoEncoder(chunk_size=1).to(device)

    # Target Latent resolution
    target_size = (360, 360)

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    images = [f for f in glob.glob(f"{scene_dir}/images/*") if f.lower().endswith(image_extensions)]
    # import ipdb; ipdb.set_trace()
    os.makedirs(f"{scene_dir}/latents", exist_ok=True)

    for image_path in tqdm(images, desc="Encoding images"):
        latent_save_path = os.path.splitext(image_path.replace("images", "latents"))[0] + ".pt"
        if os.path.exists(latent_save_path):
            continue  # Skip if latent already exists
        
        image = iio.imread(image_path)
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0  # [H, W, C]
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Create Latent Image encodigns
        encoded = AE.encode(image_tensor, 1)

        # Resize latent according to the desired shape
        encoded_resized = F.interpolate(encoded, target_size, mode='bilinear', align_corners=False)

        torch.save(encoded_resized, latent_save_path)

    del AE
    torch.cuda.empty_cache()



def resize_images(scene_dir: str, size=(360, 360)):
    # ipdb.set_trace()
    source_dir = f"{scene_dir}/images"
    target_dir = f"{scene_dir}/train_images"
    os.makedirs(target_dir, exist_ok=True)
    
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        
        if os.path.isfile(source_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            try:
                with Image.open(source_path) as img:
                    img_resized = img.resize(size, resample=Image.Resampling.LANCZOS)
                    target_path = os.path.join(target_dir, filename)
                    img_resized.save(target_path)
                    # print(f"Resized and saved: {target_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
