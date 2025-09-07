from flask import Flask, render_template, request, send_from_directory
import os
from flask import send_from_directory, jsonify
import open3d as o3d
from datetime import datetime
import cv2

from fusion.fuse_maps import fuse_point_clouds_with_init, make_se3
from fusion.fuse_png import fuse_without_gap

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
STATIC_FOLDER = 'static'
LOG_FILE = 'upload.log'

os.makedirs(os.path.join(UPLOAD_FOLDER, 'drone1'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'drone2'), exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def log_upload(filename, client_ip):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {client_ip} → {filename}\n"
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)

def ply_to_glb(ply_path, glb_path):
    # PLY üçgen mesh ise direkt; değilse point cloud için basit meshleme yap
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            raise RuntimeError("PLY boş (point cloud).")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # Ball Pivoting yedek Poisson
        try:
            r = 0.1
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([r, r*2])
            )
        except Exception:
            mesh = o3d.geometry.TriangleMesh()
        if mesh.is_empty() or len(mesh.triangles) < 50:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.compute_vertex_normals()
    os.makedirs(os.path.dirname(glb_path) or ".", exist_ok=True)
    if not o3d.io.write_triangle_mesh(glb_path, mesh):
        raise RuntimeError("GLB yazılamadı.")

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/map")
def map_view():
    return render_template("map.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    # Dosya var mı?
    if 'file' not in request.files:
        return "form-data içinde 'file' alanı yok.", 400

    file = request.files['file']
    if file.filename == '':
        return "Dosya adı boş.", 400

    filename = file.filename

    # drone1 / drone2 kontrolü
    if 'drone1' in filename:
        subfolder = 'drone1'
    elif 'drone2' in filename:
        subfolder = 'drone2'
    else:
        return "Dosya adı 'drone1' veya 'drone2' içermeli.", 400

    # Kaydet
    save_dir = os.path.join(UPLOAD_FOLDER, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    file.save(save_path)

    # Not: Burada GLB dönüşümü YOK. Hemen 200 dönüyoruz.
    return f"{filename} yüklendi.", 200

@app.route('/fusePLY', methods=['GET'])
def fuse_ply():
    # Query parametrelerini oku (yoksa 0 kabul)
    def fget(name, default=0.0):
        v = request.args.get(name, default)
        try:
            return float(v)
        except:
            return default

    dx    = fget("dx", 0.0)
    dy    = fget("dy", 0.0)
    dz    = fget("dz", 0.0)
    yaw   = fget("yaw", 0.0)    # derece
    roll  = fget("roll", 0.0)
    pitch = fget("pitch", 0.0)
    refine = (request.args.get("refine", "true").lower() != "false")  # default: true

    init_T = make_se3(dx, dy, dz, roll, pitch, yaw)

    # 1) PLY birleştir (başlangıç dönüşümü ile)
    msg, ok = fuse_point_clouds_with_init(
        pcd1_path="static/drone1/drone1_output.ply",
        pcd2_path="static/drone2/drone2_output.ply",
        voxel_size=0.05,
        init_T=init_T,
        refine_icp=refine
    )
    if not ok:
        return jsonify(error=msg), 400

    # 2) GLB üret
    ply_path = os.path.abspath(os.path.join("static", "fused_map.ply"))
    glb_path = os.path.abspath(os.path.join("static", "fused_map.glb"))
    try:
        ply_to_glb(ply_path, glb_path)
    except Exception as e:
        return jsonify(error=f"GLB dönüşüm hatası: {e}"), 500

    d, f = os.path.split(glb_path)
    resp = send_from_directory(d, f, as_attachment=False, max_age=0)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route('/fusePNG', methods=['GET'])
def fuse_png():
    # 1) PNG'yi üret
    out_path, success = fuse_without_gap(
        img1_path="static/drone1/lidar.png",
        img2_path="static/drone1/lidar.png",
        out_path="static/fused_map.png",
        direction="horizontal"
    )
    if not success:
        return out_path, 400

    # 2) Dosyayı güvenli şekilde döndür (cache kapalı)
    d, f = os.path.split(os.path.abspath(out_path))
    if not os.path.isfile(os.path.join(d, f)):
        return "Çıktı dosyası bulunamadı.", 404

    resp = send_from_directory(d, f, as_attachment=False, max_age=0)
    # Cache’i tamamen kapatmak istersen:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
