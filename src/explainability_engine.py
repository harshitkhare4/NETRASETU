"""
src/explainability_engine.py
Unified Explainability & Multi-Modal Evidence Fusion Engine for NetraSetu.

Synthesizes:
1. Quality Assessment & Quality Gate (GOOD / REVIEW / REJECT -> NEEDS RECAPTURE)
2. Primary Classifier (NetraSetu_ResNet50_best.pth)
3. Confidence Calibration (Temperature Scaling T = 0.9215)
4. Referable Risk Screening (P(ref) = P2 + P3 + P4 >= 0.24)
5. Grad-CAM visual attention
6. Anatomical Landmark Localization (Optic Disc & Fovea)
7. Lesion Segmentation (MA, HE, Hard EX, Soft EX via UNet V3)
8. Retinal Blood Vessel Segmentation (Vessel UNet)
9. Evidence Fusion (Combined multi-layer clinical explainability map)
"""

import os
import sys
import json
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Local models
sys.path.insert(0, r"C:\NetraSetu")
from src.train_idrid_localization import KeypointResNet18, extract_keypoint_from_heatmap, processed_to_original_coordinate
from src.train_unet_v3 import UNetV3
from src.train_drive_vessel import VesselUNet

class ExplainabilityEngine:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root = r"C:\NetraSetu"
        self.models_dir = os.path.join(self.root, "models")
        self.results_dir = os.path.join(self.root, "results", "final_pipeline", "explainability")

        # Class Names
        self.class_names = [
            "No DR (Grade 0)",
            "Mild DR (Grade 1)",
            "Moderate DR (Grade 2)",
            "Severe DR (Grade 3)",
            "Proliferative DR (Grade 4)"
        ]

        # 1. Load Baseline ResNet-50 Classifier
        self.classifier = models.resnet50(weights=None)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 5)
        cls_ckpt_path = os.path.join(self.models_dir, "NetraSetu_ResNet50_best.pth")
        cls_ckpt = torch.load(cls_ckpt_path, map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(cls_ckpt['model_state_dict'] if 'model_state_dict' in cls_ckpt else cls_ckpt)
        self.classifier.to(self.device).eval()

        # 2. Grad-CAM Target Layer
        self.cam = GradCAM(model=self.classifier, target_layers=[self.classifier.layer4[-1]])

        # 3. Calibration Configuration
        calib_cfg_path = os.path.join(self.root, "results", "final_pipeline", "calibration", "temperature_scaling.json")
        if os.path.exists(calib_cfg_path):
            with open(calib_cfg_path, 'r') as f:
                calib_data = json.load(f)
            self.temperature = float(calib_data.get('optimal_temperature', 0.9215))
        else:
            self.temperature = 0.9215

        # 4. Localization Model (Lazy loaded)
        self.loc_model = None

        # 5. Lesion Segmentation Model (Lazy loaded)
        self.seg_model = None
        self.seg_thresholds = {
            'Microaneurysm': 0.28,
            'Hemorrhage': 0.70,
            'Hard Exudate': 0.70,
            'Soft Exudate': 0.70
        }
        v3_thresh_path = os.path.join(self.root, "results", "final_pipeline", "segmentation", "v3_optimal_thresholds.json")
        if os.path.exists(v3_thresh_path):
            with open(v3_thresh_path, 'r') as f:
                th_data = json.load(f)
                for k in self.seg_thresholds:
                    if k in th_data:
                        self.seg_thresholds[k] = float(th_data[k]['threshold'])

        # 6. Vessel Segmentation Model (Lazy loaded)
        self.vessel_model = None

    def _ensure_loc_model(self):
        if self.loc_model is None:
            p = os.path.join(self.models_dir, "NetraSetu_IDRiD_Localization_best.pth")
            self.loc_model = KeypointResNet18(out_channels=2).to(self.device)
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            self.loc_model.load_state_dict(ckpt['model_state_dict'])
            self.loc_model.eval()

    def _ensure_seg_model(self):
        if self.seg_model is None:
            p = os.path.join(self.models_dir, "NetraSetu_IDRiD_UNet_V3_best.pth")
            self.seg_model = UNetV3(num_classes=4).to(self.device)
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            self.seg_model.load_state_dict(ckpt['model_state_dict'])
            self.seg_model.eval()

    def _ensure_vessel_model(self):
        if self.vessel_model is None:
            p = os.path.join(self.models_dir, "NetraSetu_DRIVE_Vessel_UNet_best.pth")
            self.vessel_model = VesselUNet().to(self.device)
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            self.vessel_model.load_state_dict(ckpt['model_state_dict'])
            self.vessel_model.eval()

    def assess_quality(self, image_bgr):
        """
        Calculates blur, contrast, brightness, and retinal FOV.
        Applies Quality Gate: GOOD / REVIEW / REJECT (NEEDS RECAPTURE).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        fov_ratio = float(np.count_nonzero(binary) / binary.size)

        reasons = []
        is_reject = False

        if blur < 8.0:
            reasons.append("Severe motion/optical blur")
            is_reject = True
        elif blur < 20.0:
            reasons.append("Mild/borderline blur")

        if contrast < 8.0:
            reasons.append("Severely low contrast")
            is_reject = True
        elif contrast < 15.0:
            reasons.append("Sub-optimal contrast")

        if brightness < 12.0:
            reasons.append("Severely underexposed / dark")
            is_reject = True
        elif brightness < 20.0:
            reasons.append("Low illumination")
        elif brightness > 240.0:
            reasons.append("Overexposed / glare")

        if fov_ratio < 0.10:
            reasons.append("Non-retinal or severely clipped field of view")
            is_reject = True
        elif fov_ratio < 0.15:
            reasons.append("Reduced retinal field of view")

        if is_reject:
            status = "REJECT"
        elif len(reasons) > 0:
            status = "REVIEW"
        else:
            status = "GOOD"

        return {
            'status': status,
            'blur_score': round(blur, 2),
            'contrast': round(contrast, 2),
            'brightness': round(brightness, 2),
            'retinal_area_ratio': round(fov_ratio, 4),
            'reasons': reasons,
            'needs_recapture': (status == "REJECT")
        }

    def run_inference(self, image_path, is_idrid_domain=False, is_drive_domain=False):
        """
        Full end-to-end explainability inference pipeline.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_bgr.shape[:2]

        # 1. Quality Assessment
        quality = self.assess_quality(img_bgr)
        if quality['status'] == "REJECT":
            return {
                'quality': quality,
                'status': 'NEEDS RECAPTURE',
                'reason': ", ".join(quality['reasons']),
                'action': 'Stop primary screening. Recapture retinal image before clinical triage.'
            }

        # 2. Classification & Calibration
        tf_cls = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_cls = tf_cls(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(t_cls)
            raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            calib_logits = logits / self.temperature
            calib_probs = torch.softmax(calib_logits, dim=1).cpu().numpy()[0]

        pred_grade = int(np.argmax(calib_probs))
        referable_score = float(np.sum(calib_probs[2:])) # P2 + P3 + P4
        is_referable = bool(referable_score >= 0.24)

        # 3. Grad-CAM
        targets = [ClassifierOutputTarget(pred_grade)]
        cam_map = self.cam(input_tensor=t_cls, targets=targets)[0] # (224, 224)
        cam_full = cv2.resize(cam_map, (orig_w, orig_h))
        gradcam_blend = show_cam_on_image(img_rgb.astype(np.float32) / 255.0, cam_full, use_rgb=True)

        # 4. Anatomical Landmarks
        self._ensure_loc_model()
        # Resize to 512x512 letterbox
        scale = min(512 / orig_w, 512 / orig_h)
        nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
        pad_x, pad_y = (512 - nw) / 2.0, (512 - nh) / 2.0
        resized_im = cv2.resize(img_rgb, (nw, nh))
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        canvas[int(round(pad_y)):int(round(pad_y))+nh, int(round(pad_x)):int(round(pad_x))+nw] = resized_im

        norm_canvas = (canvas.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t_loc = torch.from_numpy(norm_canvas.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                loc_heatmaps = self.loc_model(t_loc)
            hm_np = loc_heatmaps.float().cpu().numpy()[0]

        od_hm_x, od_hm_y = extract_keypoint_from_heatmap(hm_np[0], size=128, use_subpixel=True)
        fov_hm_x, fov_hm_y = extract_keypoint_from_heatmap(hm_np[1], size=128, use_subpixel=True)

        od_512_x, od_512_y = od_hm_x * 4.0, od_hm_y * 4.0
        fov_512_x, fov_512_y = fov_hm_x * 4.0, fov_hm_y * 4.0

        od_orig_x, od_orig_y = processed_to_original_coordinate(od_512_x, od_512_y, orig_w, orig_h)
        fov_orig_x, fov_orig_y = processed_to_original_coordinate(fov_512_x, fov_512_y, orig_w, orig_h)

        loc_result = {
            'optic_disc': {'x': round(od_orig_x, 1), 'y': round(od_orig_y, 1)},
            'fovea': {'x': round(fov_orig_x, 1), 'y': round(fov_orig_y, 1)},
            'domain_validated': is_idrid_domain,
            'disclaimer': "Official IDRiD localization model." if is_idrid_domain else "Cross-dataset research inference — not validated for this acquisition domain."
        }

        # 5. Lesion Segmentation (512x512)
        self._ensure_seg_model()
        norm_seg = (cv2.resize(img_rgb, (512, 512)).astype(np.float32)/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t_seg = torch.from_numpy(norm_seg.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                seg_logits = self.seg_model(t_seg)
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()[0]

        lesion_names = ['Microaneurysm', 'Hemorrhage', 'Hard Exudate', 'Soft Exudate']
        lesion_summary = {}
        lesion_masks_orig = {}

        for c, name in enumerate(lesion_names):
            th = self.seg_thresholds[name]
            mask_512 = (seg_probs[c] >= th).astype(np.uint8)
            mask_full = cv2.resize(mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            lesion_masks_orig[name] = mask_full

            pixel_count = int(np.sum(mask_full > 0))
            if pixel_count > 100:
                det_status = "detected"
            elif pixel_count > 10:
                det_status = "low-confidence"
            else:
                det_status = "not detected"

            lesion_summary[name] = {
                'status': det_status,
                'pixel_count': pixel_count,
                'threshold_applied': th
            }

        # 6. Retinal Vessel Segmentation
        self._ensure_vessel_model()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                vessel_logits = self.vessel_model(t_seg)
            vessel_probs = torch.sigmoid(vessel_logits).cpu().numpy()[0, 0]
        vessel_mask_full = cv2.resize((vessel_probs >= 0.5).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        vessel_density = float(np.sum(vessel_mask_full > 0) / (orig_w * orig_h))

        vessel_summary = {
            'available': True,
            'vessel_density': round(vessel_density, 4),
            'domain_validated': is_drive_domain,
            'disclaimer': "DRIVE vessel model." if is_drive_domain else "Research vessel segmentation — evaluated on DRIVE."
        }

        # 7. Multi-Modal Evidence Fusion Overlay
        combined = img_rgb.copy()

        # Overlay Grad-CAM faintly (alpha=0.3)
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        combined = cv2.addWeighted(combined, 0.75, cam_colored, 0.25, 0)

        # Overlay Lesions as colored contours
        # HE: Red (239, 68, 68), Hard EX: Cyan (6, 182, 212), Soft EX: Magenta (217, 70, 239), MA: Yellow (234, 179, 8)
        color_map = {
            'Microaneurysm': (234, 179, 8),
            'Hemorrhage': (239, 68, 68),
            'Hard Exudate': (6, 182, 212),
            'Soft Exudate': (217, 70, 239)
        }
        for name, mask in lesion_masks_orig.items():
            if np.sum(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(combined, contours, -1, color_map[name], 2)

        # Overlay Landmarks
        # Optic Disc: Green circle
        cv2.circle(combined, (int(od_orig_x), int(od_orig_y)), 35, (34, 197, 94), 3)
        cv2.putText(combined, "OD", (int(od_orig_x) + 40, int(od_orig_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2)

        # Fovea: Orange/Yellow cross
        fx, fy = int(fov_orig_x), int(fov_orig_y)
        cv2.drawMarker(combined, (fx, fy), (245, 158, 11), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
        cv2.putText(combined, "Fovea", (fx + 20, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 158, 11), 2)

        return {
            'quality': quality,
            'predicted_grade': pred_grade,
            'predicted_label': self.class_names[pred_grade],
            'raw_class_probabilities': [round(float(p), 4) for p in raw_probs],
            'calibrated_class_probabilities': [round(float(p), 4) for p in calib_probs],
            'calibrated_confidence': round(float(calib_probs[pred_grade]), 4),
            'referable_score': round(referable_score, 4),
            'referable_status': "REFERABLE (Risk >= 24%)" if is_referable else "NON-REFERABLE (Risk < 24%)",
            'optic_disc_prediction': loc_result['optic_disc'],
            'fovea_prediction': loc_result['fovea'],
            'localization_meta': loc_result,
            'lesion_summary': lesion_summary,
            'vessel_summary': vessel_summary,
            'images': {
                'original': img_rgb,
                'gradcam': gradcam_blend,
                'combined_evidence': combined,
                'vessel_mask': vessel_mask_full,
                'lesion_masks': lesion_masks_orig
            }
        }

if __name__ == '__main__':
    print("Testing ExplainabilityEngine initialization...")
    engine = ExplainabilityEngine()
    print("ExplainabilityEngine successfully initialized!")
