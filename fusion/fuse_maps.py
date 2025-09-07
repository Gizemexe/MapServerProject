# fusion/fuse_maps.py
import numpy as np
import open3d as o3d
import os
import math

def _prepare_pcd(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0, max_nn=30
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5.0, max_nn=100
        )
    )
    return pcd_down, fpfh

def _refine_icp(source, target, init_T, voxel_size):
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2.0, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2.0, max_nn=30))
    dist_thresh = voxel_size * 1.0
    return o3d.pipelines.registration.registration_icp(
        source, target, dist_thresh, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

def _euler_rpy_to_R(roll, pitch, yaw):
    # roll (x), pitch (y), yaw (z) – radyan
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]])
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])
    return Rz @ Ry @ Rx  # Z * Y * X

def make_se3(dx=0.0, dy=0.0, dz=0.0, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """metre ve derece ile SE(3) 4x4 dönüşüm matrisi üretir."""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    R = _euler_rpy_to_R(roll, pitch, yaw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([dx, dy, dz], dtype=np.float64)
    return T

def fuse_point_clouds_with_init(
    pcd1_path="static/drone1/drone1_output.ply",
    pcd2_path="static/drone2/drone2_output.ply",
    voxel_size=0.05,
    init_T=None,           # 4x4 numpy matrisi (SE3)
    refine_icp=True        # True ise ICP ile ince hizalama
):
    if not os.path.exists(pcd1_path) or not os.path.exists(pcd2_path):
        return "PLY dosyaları bulunamadı.", False
    try:
        pcd1 = o3d.io.read_point_cloud(pcd1_path)
        pcd2 = o3d.io.read_point_cloud(pcd2_path)
        if pcd1.is_empty() or pcd2.is_empty():
            return "Nokta bulutlarından biri boş.", False

        # Başlangıç dönüşümü verilmişse uygula
        if init_T is not None:
            pcd2.transform(init_T)

        # İsteğe bağlı ICP ince hizalama (yakınsamazsa yine de birleşim yaparız)
        if refine_icp:
            icp = _refine_icp(pcd2, pcd1, np.eye(4), voxel_size)
            # Çok düşük fitness'te (ör. 0) yine de mevcut konum korunacak
            # sadece bilgi amaçlı: istersen threshold koyup hataya düşürebilirsin
            # if icp.fitness < 0.05:
            #     return f"ICP düşük fitness: {icp.fitness:.3f}", False
            pcd2.transform(icp.transformation)

        combined = pcd1 + pcd2
        if not combined.has_colors():
            combined.paint_uniform_color([1.0, 1.0, 1.0])

        os.makedirs("static", exist_ok=True)
        out_path = "static/fused_map.ply"
        o3d.io.write_point_cloud(out_path, combined)
        return "Harita birleştirme tamamlandı (başlangıç dönüşümü ile).", True
    except Exception as e:
        return f"Birleştirme hatası: {e}", False
