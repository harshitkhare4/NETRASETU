import os
import sys
import json
import time
import uuid
import base64
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# Import tested functions directly from existing inference module
from src.inference import (
    build_model,
    assess_quality,
    crop_retina,
    enhance_image,
    is_processed_aptos,
    prepare_tensor,
    classify,
    generate_gradcam,
    create_report,
    CLASS_NAMES,
    ICDR_NAMES,
    REFERABLE_THRESHOLD,
    DEVICE,
    ROOT,
    OUTPUT_DIR,
    MODEL_PATH
)

from src.db import insert_screening, generate_case_id

# ============================================================
# NETRASETU INFERENCE SERVICE (SINGLETON / CACHED)
# ============================================================

UNET_MODEL_PATH = os.path.join(
    ROOT, "models", "NetraSetu_IDRiD_UNet_V2_best.pth"
)

LESION_NAMES = [
    "Microaneurysm",
    "Hemorrhage",
    "Hard Exudate",
    "Soft Exudate"
]

LESION_THRESHOLDS = {
    "Microaneurysm": 0.34,
    "Hemorrhage": 0.30,
    "Hard Exudate": 0.64,
    "Soft Exudate": 0.64
}

LESION_COLORS_BGR = {
    "Microaneurysm": (0, 0, 255),    # Red
    "Hemorrhage": (0, 255, 0),       # Green
    "Hard Exudate": (255, 0, 0),     # Blue
    "Soft Exudate": (0, 255, 255)    # Yellow
}

LESION_COLORS_HEX = {
    "Microaneurysm": "#ef4444",      # Red
    "Hemorrhage": "#22c55e",         # Green
    "Hard Exudate": "#3b82f6",       # Blue
    "Soft Exudate": "#eab308"        # Yellow
}


# ============================================================
# UNET V2 ARCHITECTURE (FOR EXPERIMENTAL SEGMENTATION)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)


# ============================================================
# INFERENCE SERVICE CLASS
# ============================================================

class InferenceService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InferenceService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            print("\n[NetraSetu Service] Initializing Inference Service...")
            self.device = DEVICE
            self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            print(f"[NetraSetu Service] Device: {self.device} ({self.gpu_name})")

            # Load primary ResNet-50 classifier once
            try:
                print("[NetraSetu Service] Loading ResNet-50 primary classifier...")
                self.classifier_model = build_model()
                self.classifier_loaded = True
                print("[NetraSetu Service] Primary classifier loaded into memory successfully.")
            except Exception as e:
                print(f"[NetraSetu Service] ERROR loading classifier: {e}")
                self.classifier_model = None
                self.classifier_loaded = False

            # Lazy-load holder for UNet segmentation model
            self.unet_model = None
            self.unet_loaded = False
            self._unet_lock = threading.Lock()

            self._initialized = True

    def get_system_health(self):
        """Returns structured system & model health telemetry."""
        model_exists = os.path.exists(MODEL_PATH)
        is_ready = self.classifier_loaded and model_exists
        research_mode = os.getenv("NETRASETU_RESEARCH_MODE", "false").lower() in ("true", "1", "yes")
        return {
            "status": "ready" if is_ready else "model_unavailable",
            "service": "NetraSetu",
            "mode": "research" if research_mode else "production",
            "model": os.path.basename(MODEL_PATH),
            "device": str(self.device),
            "gpu": self.gpu_name,
            "cuda_available": torch.cuda.is_available(),
            "classifier_loaded": bool(self.classifier_loaded and model_exists),
            "classifier_model_path": MODEL_PATH,
            "gradcam_available": bool(self.classifier_loaded and model_exists),
            "segmentation_available": os.path.exists(UNET_MODEL_PATH),
            "locked_referable_threshold": REFERABLE_THRESHOLD,
            "referable_threshold": REFERABLE_THRESHOLD,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def _get_unet_model(self):
        """Lazy-loads the UNet V2 lesion segmentation model on demand."""
        with self._unet_lock:
            if self.unet_loaded and self.unet_model is not None:
                return self.unet_model

            if not os.path.exists(UNET_MODEL_PATH):
                raise FileNotFoundError(f"UNet checkpoint not found at: {UNET_MODEL_PATH}")

            print("[NetraSetu Service] Lazy-loading IDRiD UNet V2 segmentation model...")
            unet = UNetV2(in_channels=3, out_channels=4)
            checkpoint = torch.load(UNET_MODEL_PATH, map_location=self.device)

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            unet.load_state_dict(state_dict, strict=True)
            unet = unet.to(self.device)
            unet.eval()

            self.unet_model = unet
            self.unet_loaded = True
            print("[NetraSetu Service] IDRiD UNet V2 model loaded successfully.")
            return self.unet_model

    def analyze(self, image_input, filename="fundus.jpg", is_already_processed=None, source_type="upload", override_quality=False):
        """
        Executes the screening pipeline with strict Quality Gate enforcement:
        - If Quality is GOOD: Proceed to full AI screening and Grad-CAM.
        - If Quality is REVIEW and not override_quality: Halt primary screening and issue NEEDS RECAPTURE.
        - If Quality is REVIEW and override_quality: Run inference with explicit Research / Demo Override warnings.
        """
        if not self.classifier_loaded or self.classifier_model is None:
            raise RuntimeError("ResNet-50 classifier is not loaded.")

        start_time = time.time()

        # --- 1. Load image ---
        is_processed = False
        source_path = None

        if isinstance(image_input, str):
            source_path = image_input
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            is_processed = is_processed_aptos(image_input)
            base_name = os.path.splitext(os.path.basename(image_input))[0]
        elif isinstance(image_input, (bytes, bytearray)):
            np_arr = np.frombuffer(image_input, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Could not decode image from provided byte stream.")
            raw_base = os.path.splitext(os.path.basename(filename))[0]
            sanitized = "".join(c for c in raw_base if c.isalnum() or c in ("-", "_")).strip() or "upload"
            base_name = f"{sanitized}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        else:
            raise TypeError("image_input must be a file path string or raw bytes.")

        if is_already_processed is not None:
            is_processed = bool(is_already_processed)

        # Generate anonymous Case ID
        case_id = generate_case_id()

        # --- 2. Quality Assessment Gate ---
        quality = assess_quality(image_bgr)
        is_quality_inadequate = (quality["status"] == "REVIEW")

        recapture_advice = None
        if is_quality_inadequate:
            reasons_text = ", ".join(quality["reasons"])
            recapture_advice = (
                f"Image quality inadequate for primary screening ({reasons_text}). "
                "Please recapture with centered retinal focus, stable illumination, and minimal eyelid obscuration."
            )

        # --- 3. Preprocessing (Double-preprocessing protection) ---
        if is_processed:
            enhanced = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA)
            preprocessing_note = "Processed dataset image detected; preserved without double-preprocessing."
        else:
            enhanced = enhance_image(image_bgr)
            preprocessing_note = "Raw fundus image: executed retinal border crop, bilateral denoising, and CLAHE illumination enhancement."

        enhanced_filename = f"{base_name}_enhanced.jpg"
        enhanced_path = os.path.join(OUTPUT_DIR, enhanced_filename)
        cv2.imwrite(enhanced_path, enhanced)

        report_filename = f"{base_name}_report.txt"
        json_filename = f"{base_name}_result.json"
        report_path = os.path.join(OUTPUT_DIR, report_filename)
        json_path = os.path.join(OUTPUT_DIR, json_filename)

        # --- QUALITY GATE ENFORCEMENT ---
        if is_quality_inadequate and not override_quality:
            # STOP PRIMARY SCREENING: Reject ungradeable images as required by PS 26038
            processing_status = "NEEDS RECAPTURE"
            reasons_str = ", ".join(quality["reasons"])
            report_text = f"""============================================================
NETRASETU RETINAL SCREENING REPORT
Demo-ready Healthcare AI Research Prototype
============================================================
Case ID:             {case_id}
Image:               {filename}
Date/Time:           {time.strftime("%Y-%m-%d %H:%M:%S")}
Source Type:         {source_type}

------------------------------------------------------------
IMAGE QUALITY ASSESSMENT GATE: REJECTED (NEEDS RECAPTURE)
------------------------------------------------------------
Quality Gate Status: INADEQUATE - PRIMARY SCREENING HALTED
Blur Score:          {quality['blur_score']:.2f} (Threshold >= 20.0)
Mean Brightness:     {quality['brightness']:.2f} (Acceptable: 20.0 - 240.0)
Contrast StdDev:     {quality['contrast']:.2f} (Threshold >= 15.0)
Retinal FOV Ratio:   {quality['retinal_area_ratio'] * 100:.1f}% (Threshold >= 15.0%)
Quality Alerts:      {reasons_str}

------------------------------------------------------------
PRIMARY SCREENING RESULT
------------------------------------------------------------
Status:              NEEDS RECAPTURE
Severity Grade:      Ungradeable - Recapture Required
Primary AI screening was STOPPED by the clinical quality gate to prevent false-negative
or false-positive classifications on ungradeable/substandard imagery.

RECAPTURE INSTRUCTIONS:
{recapture_advice}
============================================================
"""
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)

            processing_time_ms = round((time.time() - start_time) * 1000, 1)

            result_data = {
                "case_id": case_id,
                "image": filename,
                "base_name": base_name,
                "source_type": source_type,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_ms": processing_time_ms,
                "quality": quality,
                "quality_gate_passed": False,
                "is_research_override": False,
                "processing_status": "NEEDS RECAPTURE",
                "review_status": "Not Required",
                "recapture_advice": recapture_advice,
                "preprocessing_note": preprocessing_note,
                "predicted_class": None,
                "icdr_grade": "Ungradeable - Recapture Required",
                "severity": "Image Inadequate",
                "class_confidence": 0.0,
                "class_confidence_percent": 0.0,
                "class_probabilities": {},
                "class_probabilities_percent": {},
                "referable_score": 0.0,
                "referable_score_percent": 0.0,
                "referable_threshold": REFERABLE_THRESHOLD,
                "referable_threshold_percent": round(REFERABLE_THRESHOLD * 100, 2),
                "referable": False,
                "recommendation": "Recapture retinal image. Inadequate quality prevents reliable AI grading.",
                "referral_detail": f"Quality Gate Alert: {reasons_str}. Primary screening stopped to prevent misclassification.",
                "gradcam_status": "SKIPPED_QUALITY_GATE",
                "outputs": {
                    "enhanced_image": enhanced_path,
                    "gradcam": None,
                    "report": report_path,
                    "json": json_path,
                    "enhanced_url": f"/api/outputs/{enhanced_filename}",
                    "gradcam_url": None,
                    "report_download_url": f"/api/download/report/{report_filename}",
                    "json_download_url": f"/api/download/json/{json_filename}"
                },
                "disclaimer": (
                    "NetraSetu is an AI-assisted research prototype evaluated on research datasets. "
                    "It is not a clinical diagnosis and does not replace evaluation by a qualified ophthalmologist."
                )
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4)

            try:
                insert_screening(result_data)
            except Exception as db_err:
                print(f"[NetraSetu Service] Warning: Failed to insert record into DB: {db_err}")

            return result_data

        # --- 4. Prepare Tensor (224x224 ImageNet normalized) ---
        tensor = prepare_tensor(enhanced)

        # --- 5. Classification ---
        (
            predicted_class,
            probabilities,
            class_confidence,
            referable_score
        ) = classify(self.classifier_model, tensor)

        referable = bool(referable_score >= REFERABLE_THRESHOLD)
        prob_list = probabilities.detach().cpu().numpy().tolist()

        if is_quality_inadequate and override_quality:
            processing_status = "NEEDS RECAPTURE (OVERRIDDEN)"
            review_status = "Pending Review" if referable else "Not Required"
            override_warning = (
                "⚠️ RESEARCH / DEMO OVERRIDE: This image failed the clinical quality gate. "
                "Predictions on ungradeable imagery are not clinically validated."
            )
        elif referable:
            processing_status = "REQUIRES SPECIALIST REVIEW"
            review_status = "Pending Review"
            override_warning = None
        else:
            processing_status = "COMPLETED"
            review_status = "Not Required"
            override_warning = None

        if referable:
            recommendation = "Ophthalmologist review recommended."
            referral_detail = (
                f"Referable score ({referable_score * 100:.1f}%) meets or exceeds the screening threshold ({REFERABLE_THRESHOLD * 100:.0f}%). "
                "Levels 2–4 indicate referable retinopathy in this screening protocol."
            )
        else:
            recommendation = "No referral indicated by the prototype screening threshold."
            referral_detail = (
                f"Referable score ({referable_score * 100:.1f}%) is below the screening threshold ({REFERABLE_THRESHOLD * 100:.0f}%). "
                "Routine annual diabetic eye screening recommended."
            )

        # --- 6. Grad-CAM Generation ---
        gradcam_filename = f"{base_name}_gradcam.jpg"
        gradcam_path = os.path.join(OUTPUT_DIR, gradcam_filename)
        gradcam_status = "SUCCESS"
        try:
            generate_gradcam(
                self.classifier_model,
                tensor,
                predicted_class,
                enhanced,
                gradcam_path
            )
        except Exception as e:
            gradcam_status = f"FAILED: {str(e)}"
            cv2.imwrite(gradcam_path, enhanced)

        # --- 7. Create Report & JSON ---
        report_text = create_report(
            source_path or filename,
            quality,
            predicted_class,
            np.array(prob_list),
            class_confidence,
            referable_score,
            referable
        )
        if override_warning:
            report_text = f"*** {override_warning} ***\n\n" + report_text

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        processing_time_ms = round((time.time() - start_time) * 1000, 1)

        result_data = {
            "case_id": case_id,
            "image": filename,
            "base_name": base_name,
            "source_type": source_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_ms": processing_time_ms,
            "quality": quality,
            "quality_gate_passed": not is_quality_inadequate,
            "is_research_override": bool(is_quality_inadequate and override_quality),
            "override_warning": override_warning,
            "processing_status": processing_status,
            "review_status": review_status,
            "recapture_advice": recapture_advice,
            "preprocessing_note": preprocessing_note,
            "predicted_class": predicted_class,
            "icdr_grade": ICDR_NAMES[predicted_class],
            "severity": CLASS_NAMES[predicted_class],
            "class_confidence": round(class_confidence, 6),
            "class_confidence_percent": round(class_confidence * 100, 2),
            "class_probabilities": {
                CLASS_NAMES[i]: round(prob_list[i], 6)
                for i in range(5)
            },
            "class_probabilities_percent": {
                CLASS_NAMES[i]: round(prob_list[i] * 100, 2)
                for i in range(5)
            },
            "referable_score": round(referable_score, 6),
            "referable_score_percent": round(referable_score * 100, 2),
            "referable_threshold": REFERABLE_THRESHOLD,
            "referable_threshold_percent": round(REFERABLE_THRESHOLD * 100, 2),
            "referable": referable,
            "recommendation": recommendation,
            "referral_detail": referral_detail,
            "gradcam_status": gradcam_status,
            "outputs": {
                "enhanced_image": enhanced_path,
                "gradcam": gradcam_path,
                "report": report_path,
                "json": json_path,
                "enhanced_url": f"/api/outputs/{enhanced_filename}",
                "gradcam_url": f"/api/outputs/{gradcam_filename}",
                "report_download_url": f"/api/download/report/{report_filename}",
                "json_download_url": f"/api/download/json/{json_filename}"
            },
            "disclaimer": (
                "NetraSetu is an AI-assisted research prototype evaluated on research datasets. "
                "It is not a clinical diagnosis and does not replace evaluation by a qualified ophthalmologist."
            )
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4)

        # Record into SQLite database
        try:
            insert_screening(result_data)
        except Exception as db_err:
            print(f"[NetraSetu Service] Warning: Failed to insert record into DB: {db_err}")

        return result_data

    def analyze_camera_frame(self, data_url, override_quality=False):
        """Decodes base64 dataURL from browser webcam and runs analysis."""
        if not data_url or "," not in data_url:
            raise ValueError("Invalid camera data URL.")

        header, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        filename = f"webcam_capture_{int(time.time())}.jpg"
        return self.analyze(image_bytes, filename=filename, is_already_processed=False, source_type="camera", override_quality=override_quality)

    def run_experimental_lesion_analysis(self, base_name):
        """
        Executes lazy-loaded IDRiD UNet V2 lesion segmentation on the previously enhanced 512x512 image.
        """
        enhanced_path = os.path.join(OUTPUT_DIR, f"{base_name}_enhanced.jpg")
        if not os.path.exists(enhanced_path):
            raise FileNotFoundError(f"Enhanced image for '{base_name}' not found. Please analyze image first.")

        enhanced_bgr = cv2.imread(enhanced_path)
        if enhanced_bgr is None:
            raise ValueError(f"Could not read enhanced image: {enhanced_path}")

        unet = self._get_unet_model()

        rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        rgb_float = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (rgb_float - mean) / std
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = unet(tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        overlay = enhanced_bgr.copy()
        lesion_stats = {}
        total_pixels = 512 * 512

        for idx, name in enumerate(LESION_NAMES):
            thresh = LESION_THRESHOLDS[name]
            mask = probs[idx] >= thresh
            pixel_count = int(np.sum(mask))
            coverage_pct = round((pixel_count / total_pixels) * 100, 3)

            bgr_color = LESION_COLORS_BGR[name]
            overlay[mask] = bgr_color

            lesion_stats[name] = {
                "detected": bool(pixel_count > 15),
                "threshold_applied": thresh,
                "pixel_count": pixel_count,
                "area_coverage_percent": coverage_pct,
                "color_hex": LESION_COLORS_HEX[name],
                "description": self._get_lesion_description(name)
            }

        blended = cv2.addWeighted(enhanced_bgr, 0.65, overlay, 0.35, 0)

        lesions_filename = f"{base_name}_lesions.jpg"
        lesions_path = os.path.join(OUTPUT_DIR, lesions_filename)
        cv2.imwrite(lesions_path, blended)

        return {
            "base_name": base_name,
            "lesions_overlay_url": f"/api/outputs/{lesions_filename}",
            "lesion_stats": lesion_stats,
            "research_disclosure": (
                "Research visualization only. This module is an experimental research component "
                "using IDRiD-trained U-Net V2 segmentation. It is not used as the primary referral decision "
                "and has not been clinically validated (Test Mean Dice = 0.2831)."
            ),
            "benchmark_metrics": {
                "microaneurysm_test_dice": 0.0862,
                "hemorrhage_test_dice": 0.3167,
                "hard_exudate_test_dice": 0.4196,
                "soft_exudate_test_dice": 0.3098,
                "overall_mean_dice": 0.2831,
                "overall_mean_iou": 0.2025
            }
        }

    @staticmethod
    def _get_lesion_description(name):
        descriptions = {
            "Microaneurysm": "Small saccular outpouchings of retinal capillaries; earliest visible clinical sign of DR.",
            "Hemorrhage": "Bleeding within retinal layers resulting from damaged capillary walls (dot/blot or flame shaped).",
            "Hard Exudate": "Lipid and lipoprotein deposits leaking from abnormally permeable capillaries.",
            "Soft Exudate": "Cotton-wool spots representing micro-infarcts of the nerve fiber layer due to arteriolar occlusion."
        }
        return descriptions.get(name, "")


_service_instance = None

def get_inference_service():
    global _service_instance
    if _service_instance is None:
        _service_instance = InferenceService()
    return _service_instance
