
import sys
sys.path.append("/data/rohan/vggt") 

import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
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


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, depth_conf: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray, depth_conf_threshold: float = 0.5
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates with confidence thresholding.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        depth_conf (np.ndarray): Batch of depth confidence maps of shape (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)
        depth_conf_threshold (float): Threshold for valid depth confidence values

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()
    if isinstance(depth_conf, torch.Tensor):
        depth_conf = depth_conf.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, cam_coords_points, point_mask = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx], depth_conf[frame_idx], depth_conf_threshold
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_conf: np.ndarray,
    depth_conf_threshold: float = 0.5,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates with depth confidence thresholding.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        depth_conf (np.ndarray): Depth confidence map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.
        depth_conf_threshold (float): Threshold for valid depth confidence values

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None or depth_conf is None:
        return None, None, None

    # Valid depth mask based on both depth value and confidence threshold
    point_mask = (depth_map > eps) & (depth_conf > depth_conf_threshold)

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


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
        
    c2ws = torch.from_numpy(c2ws)[:, :3, :]
    Ks = torch.from_numpy(Ks)

    # # ipdb.set_trace()
    # w2cs = []
    # for i in range(c2ws.shape[0]):
    #     w2c = c2w_to_w2c(c2ws[i])
    #     w2cs.append(w2c)

    # w2cs = torch.cat(w2cs, dim=0)

    depth_conf_norm = (depth_conf - depth_conf.min()) / (depth_conf.max() - depth_conf.min())

    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                depth_conf_norm.squeeze(0),
                                                                extrinsic.squeeze(0),
                                                                intrinsic.squeeze(0),
                                                                depth_conf_threshold=0.5)

    ipdb.set_trace()

    vggt_frames = extract_camera_frames(extrinsic.squeeze(0).to(device))  # shape: [4N, 3]
    orig_frames = extract_camera_frames(c2ws.to(device))                  # shape: [4N, 3]

    T_align = umeyama_alignment_full(vggt_frames, orig_frames)

    ipdb.set_trace()

    # Step 1: Flatten the point cloud
    pts = torch.from_numpy(point_map_by_unprojection).reshape(-1, 3).to(device).float()  # shape: [6*364*364, 3]

    # Step 2: Add homogeneous coordinate
    pts_hom = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)  # shape: [N, 4]

    # Step 3: Apply alignment transformation
    pts_aligned = (T_align @ pts_hom.T).T[:, :3]  # shape: [N, 3]

    # ipdb.set_trace()
    save_pointcloud_to_ply(pts_aligned, 
                           "/data/naveen_ankit_phd/personal/SEVA-Exploration/assets/bonsai360_vggt_03/bonsai360_vggt_03/ply/vggt_05.ply")
    print("***Point Cloud Saved***")

    return pts_aligned





