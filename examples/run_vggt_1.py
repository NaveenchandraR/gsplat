
import sys
sys.path.append("/data/rohan/vggt") 

import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import torch.nn.functional as F
from torchvision import transforms as TF
import cv2
import numpy as np

import open3d as o3d
import imageio.v3 as iio
from PIL import Image
import numpy as np
import ipdb


#########################################################################################################

def resize_intrinsics_torch(K: torch.Tensor, orig_size: tuple[int, int], new_size: tuple[int, int]) -> torch.Tensor:
    """Resize intrinsics when resolution changes."""
    scale_y = new_size[0] / orig_size[0]
    scale_x = new_size[1] / orig_size[1]
    K_new = K.clone()
    K_new[:, 0, 0] *= scale_x  # fx
    K_new[:, 0, 2] *= scale_x  # cx
    K_new[:, 1, 1] *= scale_y  # fy
    K_new[:, 1, 2] *= scale_y  # cy
    return K_new


def invert_se3_torch(se3: torch.Tensor) -> torch.Tensor:
    """Invert a batch of SE(3) 3x4 matrices -> return 4x4 matrices."""
    R = se3[:, :3, :3]
    t = se3[:, :3, 3:]
    R_inv = R.transpose(1, 2)
    t_inv = -R_inv @ t
    # ipdb.set_trace()
    bottom = torch.tensor([0, 0, 0, 1.0], device=se3.device).view(1, 1, 4).repeat(se3.size(0), 1, 1)
    se3_inv = torch.cat([
        torch.cat([R_inv, t_inv], dim=-1),
        bottom
    ], dim=1)  # [B, 4, 4]
    return se3_inv


def unproject_depth_torch(depth_map: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Args:
        depth_map: [B, H, W]
        intrinsics: [B, 3, 3]

    Returns:
        cam_points: [B, H, W, 3]
    """
    B, H, W = depth_map.shape
    device = depth_map.device
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij"
    )
    ones = torch.ones_like(x)
    pixels = torch.stack((x, y, ones), dim=-1)  # [H, W, 3]
    pixels = pixels.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 3]

    K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
    pixels = pixels.view(B, -1, 3).transpose(1, 2)  # [B, 3, H*W]
    rays = K_inv @ pixels  # [B, 3, H*W]
    rays = rays.transpose(1, 2).view(B, H, W, 3)

    return rays * depth_map.unsqueeze(-1)  # [B, H, W, 3]


def unproject_pointcloud(
    depth_map: torch.Tensor,       # [B, S, H, W, 1]
    depth_conf: torch.Tensor,     # [B, S, H, W]
    intrinsics: torch.Tensor,     # [S, 3, 3]
    extrinsics: torch.Tensor,     # [S, 3, 4]
    orig_hw: tuple[int, int],
    conf_thresh: float = 0.4,
    target_hw: tuple[int, int] = None
) -> torch.Tensor:
    """
    Convert depth maps into a fused pointcloud using predefined intrinsics/extrinsics.
    Returns:
        [N, 3] tensor
    """
    B, S, H, W, _ = depth_map.shape
    device = depth_map.device
    pointclouds = []

    if target_hw is not None and (H, W) != target_hw:
        new_h, new_w = target_hw
        depth_map = F.interpolate(depth_map.view(B * S, 1, H, W), size=target_hw, mode='bilinear', align_corners=False).view(B, S, new_h, new_w, 1)
        depth_conf = F.interpolate(depth_conf.view(B * S, 1, H, W), size=target_hw, mode='bilinear', align_corners=False).view(B, S, new_h, new_w)
        H, W = new_h, new_w

    # ipdb.set_trace()

    intrinsics = intrinsics.to(device).unsqueeze(0) # [B, S, 3, 3]
    extrinsics = extrinsics.to(device).unsqueeze(0) # [B, S, 3, 4]

    # ipdb.set_trace()

    for s in range(S):
        depth = depth_map[:, s, :, :, 0]  # [B, H, W]
        conf = depth_conf[:, s, :, :]     # [B, H, W]

        mask = (depth > 0) & (conf > conf_thresh)  # [B, H, W]
        intr_s = resize_intrinsics_torch(intrinsics[:, s], orig_hw, (H, W))  # [B, 3, 3]

        # cam -> local points [B, H, W, 3]
        cam_pts = unproject_depth_torch(depth, intr_s)  # [B, H, W, 3]

        cam_pts = cam_pts.view(B, -1, 3)
        mask = mask.view(B, -1)

        # world transform
        ext = extrinsics[:, s]  # [B, 3, 4]
        ext_inv = invert_se3_torch(ext)  # [B, 4, 4]
        ones = torch.ones((B, cam_pts.shape[1], 1), device=device)
        cam_pts_homo = torch.cat([cam_pts, ones], dim=-1)  # [B, N, 4]

        world_pts = torch.bmm(cam_pts_homo, ext_inv.transpose(1, 2))[:, :, :3]  # [B, N, 3]

        for b in range(B):
            valid_pts = world_pts[b][mask[b]]
            if valid_pts.shape[0] > 0:
                pointclouds.append(valid_pts)

    if not pointclouds:
        return torch.empty((0, 3), device=device)

    return torch.cat(pointclouds, dim=0)  # [N, 3]


def save_pointcloud_to_ply(pointcloud: torch.Tensor, filename: str = "output.ply"):
    assert pointcloud.ndim == 2 and pointcloud.shape[1] == 3, "Expected pointcloud shape [N, 3]"
    
    # Convert to CPU and detach
    pointcloud_np = pointcloud.detach().cpu().float().numpy()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_np)

    # Optionally: estimate normals if needed for visualization
    # pcd.estimate_normals()

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud with {len(pointcloud_np)} points to {filename}")


def c2w_to_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, 4, 4] camera-to-world homogeneous matrices to world-to-camera.
    Args:
        c2w: torch.Tensor of shape [B, 4, 4]
    Returns:
        w2c: torch.Tensor of shape [B, 4, 4]
    """
    # ipdb.set_trace()
    c2w = c2w.unsqueeze(0)
    assert c2w.shape[-2:] == (4, 4), "Expected input shape [B, 4, 4]"

    R = c2w[:, :3, :3]        # [B, 3, 3]
    t = c2w[:, :3, 3:]        # [B, 3, 1]

    R_inv = R.transpose(1, 2)            # [B, 3, 3]
    t_inv = -torch.bmm(R_inv, t)         # [B, 3, 1]

    # Construct new [B, 4, 4] matrices
    bottom_row = torch.tensor([0, 0, 0, 1], device=c2w.device, dtype=c2w.dtype).view(1, 1, 4).expand(c2w.shape[0], -1, -1)
    w2c = torch.cat([torch.cat([R_inv, t_inv], dim=2), bottom_row], dim=1)  # [B, 4, 4]

    return w2c

#########################################################################################################

# image_paths = [
#     "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360/bonsai360/train_images/00000.png",
#     "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360/bonsai360/train_images/00011.png",
#     "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360/bonsai360/train_images/00021.png",
#     "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360/bonsai360/train_images/00031.png",
#     "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360/bonsai360/train_images/00041.png",
#     "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360/bonsai360/train_images/00051.png",
# ]

# sels = [0, 11, 21, 31, 41, 51]
# c2ws = torch.load("/data/naveen_ankit_phd/vggt_testing/vggt_gsplat/c2ws.pt")
# Ks = torch.load("/data/naveen_ankit_phd/vggt_testing/vggt_gsplat/Ks.pt")

# c2ws = c2ws[sels]
# Ks = Ks[sels]

# depth_map = torch.load("/data/naveen_ankit_phd/vggt_testing/vggt_gsplat/depth_map.pt")
# depth_conf = torch.load("/data/naveen_ankit_phd/vggt_testing/vggt_gsplat/depth_conf.pt")
# depth_conf_norm = (depth_conf - depth_conf.min()) / (depth_conf.max() - depth_conf.min())


# target_hw = (360, 360)
# orig_hw = (364, 364)

# # ipdb.set_trace()

# pointcloud = unproject_pointcloud(
#     depth_map, 
#     depth_conf_norm,
#     Ks, 
#     c2ws,
#     orig_hw=orig_hw,
#     conf_thresh=0.5,
#     target_hw=target_hw
# )
# print(pointcloud.shape)  # torch.Size([N, 3])

def run_vggt(image_paths, c2ws, Ks):

    image_arr = []
    for image_path in image_paths:
        image = iio.imread(image_path)  # shape: [H, W, C]
        image_resized = Image.fromarray(image).resize((364, 364), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image_resized)).float() / 255.0  # [H, W, C]
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_arr.append(image_tensor)

    input_images = torch.cat(image_arr, dim=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # # Initialize the model and load the pretrained weights.
    # # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    print(f"Loaded {len(input_images)} for processing")

    # images = preprocess_images(input_images, mode).to(device)
    images = input_images.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)         
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        
    c2ws = torch.from_numpy(c2ws)
    Ks = torch.from_numpy(Ks)

    # ipdb.set_trace()
    w2cs = []
    for i in range(c2ws.shape[0]):
        w2c = c2w_to_w2c(c2ws[i])
        w2cs.append(w2c)

    w2cs = torch.cat(w2cs, dim=0)

    depth_conf_norm = (depth_conf - depth_conf.min()) / (depth_conf.max() - depth_conf.min())

    target_hw = (360, 360)
    orig_hw = (364, 364)

    pointcloud = unproject_pointcloud(
        depth_map, 
        depth_conf_norm,
        Ks, 
        w2cs,
        orig_hw=orig_hw,
        conf_thresh=0.3,
        target_hw=target_hw
    )

    del model
    torch.cuda.empty_cache()

    save_pointcloud_to_ply(pointcloud, 
                           "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360_vggt_03/bonsai360_vggt_03/ply/vggt_05.ply")
    print("***Point Cloud Saved***")

    return pointcloud





