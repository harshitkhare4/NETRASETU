import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# NETRASETU - END-TO-END INFERENCE
# ============================================================

ROOT = r"C:\NetraSetu"

MODEL_PATH = os.path.join(
    ROOT, "models", "NetraSetu_ResNet50_best.pth"
)

OUTPUT_DIR = os.path.join(
    ROOT, "results", "inference"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

ICDR_NAMES = [
    "Level 0 - No DR",
    "Level 1 - Mild DR",
    "Level 2 - Moderate DR",
    "Level 3 - Severe DR",
    "Level 4 - Proliferative DR"
]

REFERABLE_THRESHOLD = 0.24


# ============================================================
# BUILD RESNET-50
# ============================================================

def build_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        5
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# QUALITY ASSESSMENT
# ============================================================

def assess_quality(image_bgr):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    _, binary = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    retinal_area_ratio = (
        np.count_nonzero(binary)
        / binary.size
    )

    reasons = []

    if blur_score < 20:
        reasons.append("Image appears blurry")

    if contrast < 15:
        reasons.append("Low image contrast")

    if brightness < 20:
        reasons.append("Image is too dark")

    if brightness > 240:
        reasons.append("Image is overexposed")

    if retinal_area_ratio < 0.15:
        reasons.append(
            "Insufficient retinal field of view"
        )

    status = (
        "GOOD"
        if len(reasons) == 0
        else "REVIEW"
    )

    return {
        "status": status,
        "blur_score": round(
            float(blur_score), 2
        ),
        "brightness": round(
            brightness, 2
        ),
        "contrast": round(
            contrast, 2
        ),
        "retinal_area_ratio": round(
            float(retinal_area_ratio), 4
        ),
        "reasons": reasons
    }


# ============================================================
# CROP BLACK BORDER
# ============================================================

def crop_retina(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest
    )

    pad = 5

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    cropped = image[y1:y2, x1:x2]

    if (
        cropped.shape[0] > 100
        and cropped.shape[1] > 100
    ):
        return cropped

    return image


# ============================================================
# ENHANCE IMAGE
# ============================================================

def enhance_image(image_bgr):

    image = crop_retina(image_bgr)

    image = cv2.resize(
        image,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    image = cv2.bilateralFilter(
        image,
        5,
        30,
        30
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([
        l_channel,
        a_channel,
        b_channel
    ])

    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ============================================================
# CHECK WHETHER IMAGE IS ALREADY PROCESSED
# ============================================================

def is_processed_aptos(path):

    normalized = os.path.normpath(
        path
    ).lower()

    return (
        "\\data\\aptos\\processed\\"
        in normalized
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_tensor(image_bgr):

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Match ResNet-50 training/evaluation input size
    rgb = cv2.resize(
        rgb,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    image = rgb.astype(
        np.float32
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = (
        image - mean
    ) / std

    image = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float()

    image = image.unsqueeze(0)

    return image.to(DEVICE)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(model, tensor):

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    class_confidence = float(
        probabilities[
            predicted_class
        ].item()
    )

    referable_score = float(
        probabilities[2:].sum().item()
    )

    return (
        predicted_class,
        probabilities,
        class_confidence,
        referable_score
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    tensor,
    predicted_class,
    image_bgr,
    output_path
):

    # --------------------------------------------------------
    # Grad-CAM operates at the model input resolution (224x224)
    # --------------------------------------------------------

    cam = GradCAM(
        model=model,
        target_layers=[model.layer4[-1]]
    )

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    grayscale_cam = cam(
        input_tensor=tensor,
        targets=targets
    )[0]

    # --------------------------------------------------------
    # Prepare original/enhanced image for final display
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    rgb = cv2.resize(
        rgb,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    rgb_float = (
        rgb.astype(np.float32) / 255.0
    )

    # --------------------------------------------------------
    # Resize CAM from 224x224 -> 512x512
    # --------------------------------------------------------

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (512, 512),
        interpolation=cv2.INTER_LINEAR
    )

    # Keep valid CAM range
    grayscale_cam = np.clip(
        grayscale_cam,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return output_path

# ============================================================
# CREATE QUALITY GATE REPORT (NEEDS RECAPTURE)
# ============================================================

def create_quality_gate_report(
    image_path,
    quality,
    recapture_advice
):

    filename = os.path.basename(
        image_path
    )

    lines = []

    lines.append(
        "========================================"
    )
    lines.append(
        "NETRASETU SCREENING REPORT"
    )
    lines.append(
        "========================================"
    )
    lines.append("")

    lines.append(
        f"Image              : {filename}"
    )

    lines.append("")
    lines.append("IMAGE QUALITY ASSESSMENT: REJECTED")

    lines.append(
        f"Status             : {quality['status']} (NEEDS RECAPTURE)"
    )

    lines.append(
        f"Blur Score         : {quality['blur_score']}"
    )

    lines.append(
        f"Brightness         : {quality['brightness']}"
    )

    lines.append(
        f"Contrast           : {quality['contrast']}"
    )

    lines.append(
        f"Retinal Area       : "
        f"{quality['retinal_area_ratio']}"
    )

    if quality["reasons"]:
        lines.append("Quality Alerts:")
        for reason in quality["reasons"]:
            lines.append(
                f"  - {reason}"
            )
    else:
        lines.append(
            "Quality Alerts     : None"
        )

    lines.append("")
    lines.append("PRIMARY SCREENING RESULT")
    lines.append(
        "Status             : NEEDS RECAPTURE"
    )
    lines.append(
        "Severity Grade     : Ungradeable - Recapture Required"
    )
    lines.append(
        "Primary AI screening was HALTED by the clinical quality gate to prevent"
    )
    lines.append(
        "false-negative or false-positive classifications on ungradeable/substandard imagery."
    )

    lines.append("")
    lines.append("RECAPTURE GUIDANCE :")
    lines.append(
        f"{recapture_advice}"
    )

    lines.append("")
    lines.append("IMPORTANT DISCLAIMER")
    lines.append(
        "This is an AI-assisted screening prototype "
        "evaluated on research datasets."
    )
    lines.append(
        "It is not a clinical diagnosis and does not "
        "replace qualified ophthalmologist review."
    )

    lines.append(
        "========================================"
    )

    return "\n".join(lines)


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(
    image_path,
    quality,
    predicted_class,
    probabilities,
    class_confidence,
    referable_score,
    referable
):

    filename = os.path.basename(
        image_path
    )

    lines = []

    lines.append(
        "========================================"
    )
    lines.append(
        "NETRASETU SCREENING REPORT"
    )
    lines.append(
        "========================================"
    )
    lines.append("")

    lines.append(
        f"Image              : {filename}"
    )

    lines.append("")
    lines.append("IMAGE QUALITY")

    lines.append(
        f"Status             : {quality['status']}"
    )

    lines.append(
        f"Blur Score         : {quality['blur_score']}"
    )

    lines.append(
        f"Brightness         : {quality['brightness']}"
    )

    lines.append(
        f"Contrast           : {quality['contrast']}"
    )

    lines.append(
        f"Retinal Area       : "
        f"{quality['retinal_area_ratio']}"
    )

    if quality["reasons"]:

        lines.append("Quality Alerts:")

        for reason in quality["reasons"]:
            lines.append(
                f"  - {reason}"
            )

    else:

        lines.append(
            "Quality Alerts     : None"
        )

    lines.append("")
    lines.append("DIABETIC RETINOPATHY GRADING")

    lines.append(
        f"ICDR Grade         : "
        f"{ICDR_NAMES[predicted_class]}"
    )

    lines.append(
        f"Severity           : "
        f"{CLASS_NAMES[predicted_class]}"
    )

    lines.append(
        f"Class Confidence   : "
        f"{class_confidence * 100:.2f}%"
    )

    lines.append("")
    lines.append("CLASS PROBABILITIES")

    for i, name in enumerate(CLASS_NAMES):

        lines.append(
            f"{name:<20}: "
            f"{probabilities[i] * 100:.2f}%"
        )

    lines.append("")
    lines.append("REFERABLE DR SCREENING")

    lines.append(
        f"Referable Score    : "
        f"{referable_score * 100:.2f}%"
    )

    lines.append(
        f"Threshold          : "
        f"{REFERABLE_THRESHOLD:.2f}"
    )

    lines.append(
        f"Referable Status   : "
        f"{'YES' if referable else 'NO'}"
    )

    if referable:

        lines.append(
            "Recommendation     : "
            "Ophthalmologist review recommended."
        )

    else:

        lines.append(
            "Recommendation     : "
            "No referral indicated by the prototype threshold."
        )

    lines.append("")
    lines.append("EXPLAINABILITY")
    lines.append(
        "Grad-CAM generated for the predicted DR class."
    )

    lines.append("")
    lines.append("IMPORTANT DISCLAIMER")
    lines.append(
        "This is an AI-assisted screening prototype "
        "evaluated on research datasets."
    )
    lines.append(
        "It is not a clinical diagnosis and does not "
        "replace qualified ophthalmologist review."
    )

    lines.append(
        "========================================"
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENT
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            r'python src\inference.py "path_to_image"'
        )
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):

        print()
        print(
            "ERROR: Image file not found:"
        )
        print(image_path)
        return

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("============================================")
    print("NETRASETU AI SCREENING")
    print("============================================")

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Loading ResNet-50...")

    try:

        model = build_model()

    except Exception as e:

        print()
        print(
            "ERROR while loading model:"
        )
        print(str(e))
        return

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_bgr = cv2.imread(
        image_path
    )

    if image_bgr is None:

        print()
        print(
            "ERROR: Could not read image."
        )
        return

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    print()
    print(
        "Assessing image quality..."
    )

    quality = assess_quality(
        image_bgr
    )

    print(
        f"Quality: {quality['status']}"
    )

    print(
        f"Blur score: {quality['blur_score']}"
    )

    print(
        f"Brightness: {quality['brightness']}"
    )

    print(
        f"Contrast: {quality['contrast']}"
    )

    # --------------------------------------------------------
    # QUALITY GATE ENFORCEMENT
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    report_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_report.txt"
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_result.json"
    )

    override_quality = (
        "--override-quality" in sys.argv
        or "--analyze-anyway" in sys.argv
    )

    if quality["status"] == "REVIEW" and not override_quality:
        reasons_str = ", ".join(quality["reasons"]) if quality["reasons"] else "None"
        recapture_advice = (
            f"Image quality inadequate for primary screening ({reasons_str}). "
            "Please recapture with centered retinal focus, stable illumination, and minimal eyelid obscuration."
        )

        print()
        print("============================================")
        print("QUALITY GATE STATUS : NEEDS RECAPTURE")
        print("============================================")
        print(f"Quality Status     : {quality['status']}")
        print(f"Quality Alerts     : {reasons_str}")
        print("Primary Screening  : HALTED BY QUALITY GATE")
        print("Case Status        : NEEDS RECAPTURE")
        print("Severity Grade     : Ungradeable - Recapture Required")
        print()
        print("RECAPTURE GUIDANCE :")
        print(recapture_advice)
        print()
        print("Notice: To override for research/demo purposes, pass --override-quality.")
        print("============================================")

        report = create_quality_gate_report(
            image_path,
            quality,
            recapture_advice
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(report)

        result = {
            "image": os.path.basename(image_path),
            "quality": quality,
            "quality_gate_passed": False,
            "is_research_override": False,
            "processing_status": "NEEDS RECAPTURE",
            "recapture_advice": recapture_advice,
            "predicted_class": None,
            "icdr_grade": "Ungradeable - Recapture Required",
            "severity": "Image Inadequate",
            "class_confidence": 0.0,
            "class_probabilities": {},
            "referable_score": 0.0,
            "referable_threshold": REFERABLE_THRESHOLD,
            "referable": False,
            "recommendation": "Recapture retinal image. Inadequate quality prevents reliable AI grading.",
            "gradcam_status": "SKIPPED_QUALITY_GATE",
            "outputs": {
                "enhanced_image": None,
                "gradcam": None,
                "report": report_path,
                "json": json_path
            },
            "prototype_note": "Research prototype. Not a clinical diagnosis."
        }

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                result,
                f,
                indent=4
            )

        print()
        print("Outputs:")
        print(report_path)
        print(json_path)
        print("============================================")
        return

    if quality["status"] == "REVIEW" and override_quality:
        print()
        print("============================================")
        print("WARNING: RESEARCH / DEMO OVERRIDE ACTIVE")
        print("============================================")
        print("Image failed quality gate, but proceeding with AI screening as requested.")
        print("============================================")

    # --------------------------------------------------------
    # PREPROCESS ONCE
    # --------------------------------------------------------

    print()

    if is_processed_aptos(image_path):

        print(
            "Input is an already-processed APTOS image."
        )

        print(
            "Skipping second preprocessing."
        )

        enhanced = cv2.resize(
            image_bgr,
            (512, 512),
            interpolation=cv2.INTER_AREA
        )

    else:

        print(
            "Enhancing retinal image..."
        )

        enhanced = enhance_image(
            image_bgr
        )

    # --------------------------------------------------------
    # TENSOR
    # --------------------------------------------------------

    tensor = prepare_tensor(
        enhanced
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    print()
    print(
        "Running DR classification..."
    )

    (
        predicted_class,
        probabilities,
        class_confidence,
        referable_score
    ) = classify(
        model,
        tensor
    )

    referable = (
        referable_score
        >= REFERABLE_THRESHOLD
    )

    print()
    print("============================================")

    print(
        f"Predicted Grade   : "
        f"{ICDR_NAMES[predicted_class]}"
    )

    print(
        f"Severity          : "
        f"{CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Class Confidence  : "
        f"{class_confidence * 100:.2f}%"
    )

    print(
        f"Referable Score   : "
        f"{referable_score * 100:.2f}%"
    )

    print(
        f"Threshold         : "
        f"{REFERABLE_THRESHOLD:.2f}"
    )

    print(
        f"Referable         : "
        f"{'YES' if referable else 'NO'}"
    )

    print(
        "============================================"
    )

    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    enhanced_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_enhanced.jpg"
    )

    gradcam_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_gradcam.jpg"
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_report.txt"
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_result.json"
    )

    # --------------------------------------------------------
    # SAVE ENHANCED
    # --------------------------------------------------------

    cv2.imwrite(
        enhanced_path,
        enhanced
    )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    print()
    print(
        "Generating Grad-CAM..."
    )

    try:

        generate_gradcam(
            model,
            tensor,
            predicted_class,
            enhanced,
            gradcam_path
        )

        gradcam_status = "SUCCESS"

    except Exception as e:

        gradcam_status = (
            "FAILED: " + str(e)
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = create_report(
        image_path,
        quality,
        predicted_class,
        probabilities.detach().cpu().numpy(),
        class_confidence,
        referable_score,
        referable
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    result = {

        "image": os.path.basename(
            image_path
        ),

        "quality": quality,

        "predicted_class":
            predicted_class,

        "icdr_grade":
            ICDR_NAMES[predicted_class],

        "severity":
            CLASS_NAMES[predicted_class],

        "class_confidence":
            round(
                class_confidence,
                6
            ),

        "class_probabilities": {
            CLASS_NAMES[i]:
                round(
                    float(
                        probabilities[i].item()
                    ),
                    6
                )
            for i in range(5)
        },

        "referable_score":
            round(
                referable_score,
                6
            ),

        "referable_threshold":
            REFERABLE_THRESHOLD,

        "referable":
            bool(referable),

        "gradcam_status":
            gradcam_status,

        "outputs": {
            "enhanced_image":
                enhanced_path,

            "gradcam":
                gradcam_path,

            "report":
                report_path,

            "json":
                json_path
        },

        "prototype_note":
            "Research prototype. Not a clinical diagnosis."
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("============================================")
    print("NETRASETU SCREENING COMPLETE")
    print("============================================")

    print(
        f"Quality       : {quality['status']}"
    )

    print(
        f"DR Grade      : "
        f"{ICDR_NAMES[predicted_class]}"
    )

    print(
        f"Confidence    : "
        f"{class_confidence * 100:.2f}%"
    )

    print(
        f"Referable     : "
        f"{'YES' if referable else 'NO'}"
    )

    print()
    print(
        f"Grad-CAM      : {gradcam_status}"
    )

    print()
    print("Outputs:")
    print(enhanced_path)
    print(gradcam_path)
    print(report_path)
    print(json_path)

    print(
        "============================================"
    )


if __name__ == "__main__":
    main()