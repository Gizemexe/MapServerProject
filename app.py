from flask import Flask, render_template, request, send_from_directory
import os
import open3d as o3d
from datetime import datetime
from fusion.fuse_maps import fuse_point_clouds

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

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/map")
def map_view():
    return render_template("map.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = file.filename

    # Drone ID'ye göre alt klasör belirle (örneğin: drone1_output.ply)
    if 'drone1' in filename:
        subfolder = 'drone1'
    elif 'drone2' in filename:
        subfolder = 'drone2'
    else:
        return "Dosya adı drone1 veya drone2 içermeli.", 400

    save_dir = os.path.join(UPLOAD_FOLDER, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    file.save(save_path)

    # Logla
    client_ip = request.remote_addr or 'unknown'
    log_upload(filename, client_ip)

    # .ply dosyasını .glb'ye çevir
    if filename.endswith(".ply"):
        try:
            mesh = o3d.io.read_triangle_mesh(save_path)
            mesh.compute_vertex_normals()
            glb_path = os.path.join(save_dir, filename.replace(".ply", ".glb"))
            o3d.io.write_triangle_mesh(glb_path, mesh, write_ascii=False)
            print(f"{glb_path} başarıyla oluşturuldu.")
        except Exception as e:
            print(f"GLB dönüşüm hatası: {e}")

    return f"{filename} yüklendi ve işlendi.", 200


@app.route('/fuse', methods=['POST'])
def fuse():
    message, success = fuse_point_clouds()
    if not success:
        return message, 400

    # fused_map.ply → fused_map.glb dönüşümü
    ply_path = os.path.join(STATIC_FOLDER, 'fused_map.ply')
    glb_path = os.path.join(STATIC_FOLDER, 'fused_map.glb')

    try:
        mesh = o3d.io.read_triangle_mesh(ply_path)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(glb_path, mesh)
        print("✅ fused_map.glb başarıyla oluşturuldu.")
    except Exception as e:
        return f"GLB dönüşüm hatası: {e}", 500

    return "Haritalar birleştirildi ve GLB üretildi.", 200


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
