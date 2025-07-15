from flask import Flask, render_template, request, send_from_directory
import os
import open3d as o3d
from fusion.fuse_maps import fuse_point_clouds

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

os.makedirs(os.path.join(UPLOAD_FOLDER, 'drone1'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'drone2'), exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


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

    # Hangi drone'dan geldiğini belirle
    if 'drone1' in filename:
        drone_folder = os.path.join(UPLOAD_FOLDER, 'drone1')
    elif 'drone2' in filename:
        drone_folder = os.path.join(UPLOAD_FOLDER, 'drone2')
    else:
        return "Dosya adında drone1 veya drone2 geçmelidir.", 400

    save_path = os.path.join(drone_folder, filename)
    file.save(save_path)

    # .ply dosyasıysa glb üret
    if filename.endswith(".ply"):
        try:
            mesh = o3d.io.read_triangle_mesh(save_path)
            mesh.compute_vertex_normals()

            # Her drone için ayrı glb ismi
            glb_name = filename.replace(".ply", ".glb")
            glb_path = os.path.join(STATIC_FOLDER, glb_name)
            o3d.io.write_triangle_mesh(glb_path, mesh)
            print(f"{glb_name} başarıyla glb'ye çevrildi.")
        except Exception as e:
            print(f"GLB dönüşüm hatası: {e}")

    return f"{filename} başarıyla yüklendi ve işlendi.", 200


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
