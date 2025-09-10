from flask import Flask, render_template, request, send_from_directory
import os
from flask import send_from_directory, jsonify
import open3d as o3d
from datetime import datetime
import re

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

import open3d as o3d
import os

import open3d as o3d
import os


def ply_to_glb(ply_path, glb_path):
    # Önce üçgen mesh mi kontrol et
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        # Point cloud ise yükle
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            raise RuntimeError("PLY dosyası boş.")

        # Normalleri tahmin et
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Ball Pivoting ile yüzey oluştur
        try:
            r = 0.1
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([r, r * 2])
            )
        except Exception:
            mesh = o3d.geometry.TriangleMesh()

        # Ball Pivoting başarısızsa Poisson dene
        if mesh.is_empty() or len(mesh.triangles) < 50:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())

    # Normalleri hesapla
    mesh.compute_vertex_normals()

    # ⚠️ Renk bilgisi yoksa mor renkle boyayalım (RGB: 0.5, 0.1, 0.7)
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.5, 0.1, 0.7])

    # Dizin yoksa oluştur
    os.makedirs(os.path.dirname(glb_path) or ".", exist_ok=True)

    # .glb olarak kaydet
    if not o3d.io.write_triangle_mesh(glb_path, mesh, write_ascii=False, compressed=True):
        raise RuntimeError("GLB dosyası yazılamadı.")


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
    ply_path = os.path.join("static", "drone1", "drone1_output.ply")
    glb_path = os.path.join("static", "fused_map.glb")

    try:
        ply_to_glb(ply_path, glb_path)
    except Exception as e:
        return jsonify(error=f"GLB dönüşüm hatası: {str(e)}"), 500

    return send_from_directory("static", "fused_map.glb")

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

def _find_existing_radar_file(drone_no: str):
    candidates = [
        f"static/drone{drone_no}/drone{drone_no}_detected_objects.txt",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def _parse_radar_series(file_path: str):
    """
    Satır başına:
      - 'timestamp, value'  veya  'timestamp value'  veya  '... 0.35 ... 1.00 ...'
      - Hiç timestamp yoksa: t = 0,1,2,...; value = satırdaki ilk sayı
    """
    labels, values = [], []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    t_auto = 0
    for ln in lines:
        nums = re.findall(r"-?\d+(?:\.\d+)?", ln)  # satırdaki tüm sayıları yakala
        if len(nums) >= 2:
            t = float(nums[0])
            v = float(nums[1])
        elif len(nums) == 1:
            t = float(t_auto)
            v = float(nums[0])
        else:
            # sayı yoksa es geç
            continue
        labels.append(t)
        values.append(v)
        t_auto += 1

    return labels, values

@app.get("/radar/series")
def radar_series():
    drone = request.args.get("drone", "1")
    path = _find_existing_radar_file(drone)
    if not path:
        return jsonify(error=f"Radar dosyası bulunamadı (drone{drone})."), 404

    labels, values = _parse_radar_series(path)
    if not values:
        return jsonify(error="Radar verisi boş."), 204

    # İsteğe bağlı: zaman eksenini saniye formatlı stringe çevir
    labels_fmt = [str(x) for x in labels]
    return jsonify(
        ok=True,
        drone=drone,
        labels=labels_fmt,
        values=values,
        source=path
    )



@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
