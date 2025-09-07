import cv2
import numpy as np
import os

def trim_white(img, thresh=250):
    """Beyaz zeminli PNG’yi kırpar."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < thresh
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img  # tamamen beyazsa
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def fuse_without_gap(img1_path, img2_path, out_path, direction="horizontal"):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        return "Görseller okunamadı", False

    # Beyazları kırp
    img1_trim = trim_white(img1)
    img2_trim = trim_white(img2)

    if direction == "horizontal":
        # Yükseklik eşitle
        h = max(img1_trim.shape[0], img2_trim.shape[0])
        img1_resized = cv2.resize(img1_trim, (int(img1_trim.shape[1] * h / img1_trim.shape[0]), h))
        img2_resized = cv2.resize(img2_trim, (int(img2_trim.shape[1] * h / img2_trim.shape[0]), h))
        fused = cv2.hconcat([img1_resized, img2_resized])
    else:  # vertical
        # Genişlik eşitle
        w = max(img1_trim.shape[1], img2_trim.shape[1])
        img1_resized = cv2.resize(img1_trim, (w, int(img1_trim.shape[0] * w / img1_trim.shape[1])))
        img2_resized = cv2.resize(img2_trim, (w, int(img2_trim.shape[0] * w / img2_trim.shape[1])))
        fused = cv2.vconcat([img1_resized, img2_resized])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, fused)
    return out_path, True
