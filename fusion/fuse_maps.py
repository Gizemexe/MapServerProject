# fusion/fuse_maps.py
import numpy as np
import open3d as o3d
import os

MAX_TIME_DIFF = 1.0

def fuse_point_clouds():
    pcd1_path = "uploads/drone1/output.ply"
    ts1_path = "uploads/drone1/output.timestamp"

    pcd2_path = "uploads/drone2/output.ply"
    ts2_path = "uploads/drone2/output.timestamp"

    # Timestamp kontrolü
    if not os.path.exists(ts1_path) or not os.path.exists(ts2_path):
        return "Timestamp dosyaları eksik", False

    try:
        with open(ts1_path, "r") as f1, open(ts2_path, "r") as f2:
            t1 = float(f1.read().strip())
            t2 = float(f2.read().strip())
    except Exception as e:
        return f"Zaman okuma hatası: {e}", False

    if abs(t1 - t2) > MAX_TIME_DIFF:
        return "Zaman farkı çok büyük", False

    class TimeKalman:
        def __init__(self):
            self.estimated_t1 = t1
            self.estimated_t2 = t2
            self.P = np.eye(2) * 0.1
            self.Q = np.eye(2) * 0.01
            self.R = np.eye(2) * 0.1

        def update(self, z1, z2):
            z = np.array([z1, z2])
            x = np.array([self.estimated_t1, self.estimated_t2])
            y = z - x
            K = self.P @ np.linalg.inv(self.P + self.R)
            x = x + K @ y
            self.P = (np.eye(2) - K) @ self.P + self.Q
            self.estimated_t1, self.estimated_t2 = x
            return x

    kalman = TimeKalman()
    t1_adj, t2_adj = kalman.update(t1, t2)

    # Nokta bulutlarını oku
    pcd1 = o3d.io.read_point_cloud(pcd1_path)
    pcd2 = o3d.io.read_point_cloud(pcd2_path)

    # Filtreleme ve hizalama
    voxel_size = 0.05
    pcd1_down = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size=voxel_size)

    threshold = 1.0
    trans_init = np.eye(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2_down, pcd1_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    pcd2.transform(reg_p2p.transformation)
    combined = pcd1 + pcd2

    # Kaydet
    os.makedirs("static", exist_ok=True)
    out_path = "static/fused_map.ply"
    o3d.io.write_point_cloud(out_path, combined)

    return "Harita birleştirme tamamlandı.", True
