
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

def extract_camera_frames(c2ws: torch.Tensor) -> torch.Tensor:
    """
    Extracts the camera X, Y, Z axes and origin from a [N, 3, 4] camera-to-world matrix.
    Returns a [4N, 3] tensor suitable for rigid alignment.
    """
    x_axis = c2ws[:, :, 0]  # [N, 3]
    y_axis = c2ws[:, :, 1]
    z_axis = c2ws[:, :, 2]
    origin = c2ws[:, :, 3]
    return torch.cat([x_axis, y_axis, z_axis, origin], dim=0)  # [4N, 3]

def umeyama_alignment_full(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Rigid transform from A → B using full frame info (no scaling).
    A, B: [N, 3]
    Returns: [4, 4] transformation matrix
    """
    mean_A = A.mean(0)
    mean_B = B.mean(0)
    A_demean = A - mean_A
    B_demean = B - mean_B

    cov = B_demean.T @ A_demean / A.shape[0]
    U, _, Vt = torch.linalg.svd(cov)
    R = U @ Vt

    # Ensure right-handed coordinate system
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = mean_B - R @ mean_A

    T = torch.eye(4, dtype=A.dtype, device=A.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


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


#########################################################################################################


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
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))
        
    c2ws = torch.from_numpy(c2ws)[:, :3, :]
    Ks = torch.from_numpy(Ks)

    # ipdb.set_trace()

    vggt_frames = extract_camera_frames(extrinsic.squeeze(0).to(device))  # shape: [4N, 3]
    orig_frames = extract_camera_frames(c2ws.to(device))                  # shape: [4N, 3]

    T_align = umeyama_alignment_full(vggt_frames, orig_frames)

    # ipdb.set_trace()

    # Step 1: Flatten the point cloud
    pts = torch.from_numpy(point_map_by_unprojection).reshape(-1, 3).to(device).float()  # shape: [6*364*364, 3]

    # Step 2: Add homogeneous coordinate
    pts_hom = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)  # shape: [N, 4]

    # Step 3: Apply alignment transformation
    pts_aligned = (T_align @ pts_hom.T).T[:, :3]  # shape: [N, 3]

    # ipdb.set_trace()
    save_pointcloud_to_ply(pts_aligned, 
                           "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360_rotate2/bonsai360_rotate2/ply/vggt.ply")
    print("***Point Cloud Saved***")

    return pts_aligned





