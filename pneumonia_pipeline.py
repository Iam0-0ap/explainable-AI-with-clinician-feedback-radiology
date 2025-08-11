import os
import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Config ---
VALID_CUES = {
    "lower_lung": "Lobar opacity",
    "upper_lung": "Interstitial markings",
    "pleura": "Pleural effusion",
    "diffuse": "Diffuse lung opacity"
}

# Thresholds
PROBABILITY_KEEP_THRESHOLD = 0.6
HEATMAP_REGION_THRESHOLD = 0.6
DIFFUSE_COVERAGE_RATIO = 0.7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helpers ---
def load_and_preprocess(img_path, res=224):
    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 3:
        img = img.mean(2)
    img = img[None, ...]
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(res)
    ])
    img = transform(img)
    img_tensor = torch.from_numpy(img).unsqueeze(0).float()
    return img_tensor, img


def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_pneumonia_prob(model, img_tensor):
    with torch.no_grad():
        tensor = img_tensor.to(DEVICE)
        output = model(tensor)[0]
        probs = torch.sigmoid(output).cpu().numpy()
    pneumonia_idx = model.pathologies.index("Pneumonia")
    return float(probs[pneumonia_idx])


def run_gradcam(model, img_tensor, target_label="Pneumonia"):
    target_idx = model.pathologies.index(target_label)
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_idx)]
    grayscale_cam = cam(input_tensor=img_tensor.to(DEVICE), targets=targets)
    heatmap = grayscale_cam[0]
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


def map_heatmap_to_cues(heatmap, mask_threshold=HEATMAP_REGION_THRESHOLD, diffuse_ratio=DIFFUSE_COVERAGE_RATIO):
    h, w = heatmap.shape
    mask = heatmap > mask_threshold
    active_pixels = mask.sum()

    # Fallback if no strong activation
    if active_pixels == 0:
        return [VALID_CUES["diffuse"]]

    upper = mask[:h//2, :]
    lower = mask[h//2:, :]
    left = mask[:, :w//2]
    right = mask[:, w//2:]

    upper_frac = upper.sum() / active_pixels
    lower_frac = lower.sum() / active_pixels
    left_frac = left.sum() / active_pixels
    right_frac = right.sum() / active_pixels

    cues = []
    # Diffuse finding
    if max(upper_frac, lower_frac) < diffuse_ratio and (upper_frac > 0.2 and lower_frac > 0.2):
        cues.append(VALID_CUES["diffuse"])
        return cues

    # Localized lower lung
    if lower_frac >= 0.6:
        lower_left = mask[h//2:, :w//2].sum()
        lower_right = mask[h//2:, w//2:].sum()
        if lower_left / max(1, lower_left + lower_right) > 0.6:
            cues.append(f"{VALID_CUES['lower_lung']} - left lower lung")
        elif lower_right / max(1, lower_left + lower_right) > 0.6:
            cues.append(f"{VALID_CUES['lower_lung']} - right lower lung")
        else:
            cues.append(f"{VALID_CUES['lower_lung']} - bilateral lower lungs")

    # Localized upper lung
    elif upper_frac >= 0.6:
        upper_left = mask[:h//2, :w//2].sum()
        upper_right = mask[:h//2, w//2:].sum()
        if upper_left / max(1, upper_left + upper_right) > 0.6:
            cues.append(f"{VALID_CUES['upper_lung']} - left upper lung")
        elif upper_right / max(1, upper_left + upper_right) > 0.6:
            cues.append(f"{VALID_CUES['upper_lung']} - right upper lung")
        else:
            cues.append(f"{VALID_CUES['upper_lung']} - bilateral upper lungs")
    else:
        cues.append(VALID_CUES["diffuse"])

    # Pleural edge check
    border_margin = int(0.1 * w)
    left_border_mask = mask[:, :border_margin]
    right_border_mask = mask[:, -border_margin:]
    if left_border_mask.sum() / active_pixels > 0.15 or right_border_mask.sum() / active_pixels > 0.15:
        cues.append(VALID_CUES["pleura"])

    return cues


def overlay_and_save(visual_img_rgb, heatmap, out_path):
    vis = show_cam_on_image(visual_img_rgb, heatmap, use_rgb=True)
    plt.imsave(out_path, vis)


# --- Main pipeline ---
def predict_and_explain_pneumonia(img_path, model, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    img_tensor, img_numpy = load_and_preprocess(img_path, res=224)
    visual_img_rgb = np.stack([img_numpy[0], img_numpy[0], img_numpy[0]], axis=-1)
    visual_img_rgb = (visual_img_rgb - visual_img_rgb.min()) / (visual_img_rgb.max() - visual_img_rgb.min())

    # model = load_model()
    prob = predict_pneumonia_prob(model, img_tensor)

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    if prob < PROBABILITY_KEEP_THRESHOLD:
        return {
            "input_image": img_path,
            "pneumonia_probability": prob,
            "explanations": [
                {
                "label": "Not Pneumonia",
                "probability": prob,
                "cues": "Not Available",
                "heatmap_path": os.path.join(out_dir, f"{base_name}.png")
            }
            ]
        }

    heatmap = run_gradcam(model, img_tensor, target_label="Pneumonia")
    cues = map_heatmap_to_cues(heatmap)

    
    heatmap_path = os.path.join(out_dir, f"{base_name}_Pneumonia_cam.png")
    overlay_and_save(visual_img_rgb, heatmap, heatmap_path)

    return {
        "input_image": img_path,
        "pneumonia_probability": prob,
        "explanations": [
            {
                "label": "Pneumonia",
                "probability": prob,
                "cues": cues,
                "heatmap_path": heatmap_path
            }
        ]
    }


# --- For testing ---
# if __name__ == "__main__":
#     test_image = "test/00000001_000.png"
#     model = load_model()
#     result = predict_and_explain_pneumonia(test_image, model)
#     from pprint import pprint
#     pprint(result)
