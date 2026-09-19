import os
import sys
import cv2
import numpy as np

import torch
import torch.nn as nn

from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# NETRASETU - GRAD-CAM EXPLAINABLE AI
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\NetraSetu"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "NetraSetu_ResNet50_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "gradcam"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. CHECK IMAGE ARGUMENT
# ------------------------------------------------------------

if len(sys.argv) < 2:

    print("\nERROR: Image path is missing.")

    print("\nExample:")
    print(
        r'python src\gradcam.py "C:\NetraSetu\data\APTOS\processed\test\2\image.jpg"'
    )

    sys.exit(1)


IMAGE_PATH = sys.argv[1]


# ------------------------------------------------------------
# 3. CHECK IMAGE EXISTS
# ------------------------------------------------------------

if not os.path.exists(IMAGE_PATH):

    print("\nERROR: Image not found:")
    print(IMAGE_PATH)

    sys.exit(1)


# ------------------------------------------------------------
# 4. DEVICE
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\n============================================")
print("NETRASETU GRAD-CAM")
print("============================================")

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ------------------------------------------------------------
# 5. LOAD ORIGINAL IMAGE
# ------------------------------------------------------------

original_bgr = cv2.imread(
    IMAGE_PATH
)

if original_bgr is None:

    print("\nERROR: Could not read image.")

    sys.exit(1)


original_rgb = cv2.cvtColor(
    original_bgr,
    cv2.COLOR_BGR2RGB
)


# ------------------------------------------------------------
# 6. TRANSFORM
# ------------------------------------------------------------

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


input_tensor = transform(
    original_rgb
).unsqueeze(0).to(device)


# ------------------------------------------------------------
# 7. LOAD RESNET-50
# ------------------------------------------------------------

print("\nLoading model...")

model = models.resnet50(
    weights=None
)

# 5 DR classes
model.fc = nn.Linear(
    model.fc.in_features,
    5
)


# ------------------------------------------------------------
# 8. LOAD TRAINED WEIGHTS
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model file not found:")

    print(MODEL_PATH)

    sys.exit(1)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(device)

model.eval()


print("Model loaded successfully.")


# ------------------------------------------------------------
# 9. DR CLASS NAMES
# ------------------------------------------------------------

class_names = {

    0: "No DR",

    1: "Mild DR",

    2: "Moderate DR",

    3: "Severe DR",

    4: "Proliferative DR"
}


# ------------------------------------------------------------
# 10. PREDICTION
# ------------------------------------------------------------

with torch.no_grad():

    output = model(
        input_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]


predicted_class = torch.argmax(
    probabilities
).item()


predicted_name = class_names[
    predicted_class
]


class_confidence = probabilities[
    predicted_class
].item()


# ------------------------------------------------------------
# 11. REFERABLE DR SCORE
#
# Level 2 + Level 3 + Level 4
# ------------------------------------------------------------

referable_score = (
    probabilities[2]
    +
    probabilities[3]
    +
    probabilities[4]
).item()


# Same threshold selected from validation
REFERABLE_THRESHOLD = 0.24


is_referable = (
    referable_score >= REFERABLE_THRESHOLD
)


# ------------------------------------------------------------
# 12. PRINT PREDICTION
# ------------------------------------------------------------

print("\n============================================")
print("PREDICTION")
print("============================================")

print(
    f"Predicted Level: {predicted_class}"
)

print(
    f"DR Grade: {predicted_name}"
)

print(
    f"Class Confidence: "
    f"{class_confidence * 100:.2f}%"
)

print(
    f"Referable Score: "
    f"{referable_score * 100:.2f}%"
)

print(
    f"Referable Threshold: "
    f"{REFERABLE_THRESHOLD * 100:.2f}%"
)

print(
    "Referable DR:",
    "YES" if is_referable else "NO"
)


# ------------------------------------------------------------
# 13. CREATE GRAD-CAM
# ------------------------------------------------------------

print("\nCreating Grad-CAM...")


# Last convolutional layer of ResNet-50
target_layers = [
    model.layer4[-1]
]


cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# IMPORTANT:
# New Grad-CAM API requires targets

targets = [
    ClassifierOutputTarget(
        predicted_class
    )
]


grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]


# ------------------------------------------------------------
# 14. PREPARE IMAGE
# ------------------------------------------------------------

display_image = cv2.resize(
    original_rgb,
    (224, 224)
)


display_image = (
    display_image.astype(
        np.float32
    )
    / 255.0
)


# ------------------------------------------------------------
# 15. CREATE HEATMAP
# ------------------------------------------------------------

cam_image = show_cam_on_image(
    display_image,
    grayscale_cam,
    use_rgb=True
)


# Convert RGB → BGR for OpenCV saving
result = cv2.cvtColor(
    cam_image,
    cv2.COLOR_RGB2BGR
)


# ------------------------------------------------------------
# 16. ADD INFORMATION TO IMAGE
# ------------------------------------------------------------

text1 = (
    f"DR Level: "
    f"{predicted_class} - "
    f"{predicted_name}"
)

text2 = (
    f"Confidence: "
    f"{class_confidence * 100:.1f}%"
)

text3 = (
    f"Referable Score: "
    f"{referable_score * 100:.1f}%"
)

text4 = (
    "REFERABLE DR"
    if is_referable
    else
    "NON-REFERABLE DR"
)


# Add text background
cv2.rectangle(
    result,
    (0, 0),
    (224, 115),
    (0, 0, 0),
    -1
)


# ------------------------------------------------------------
# 17. WRITE TEXT
# ------------------------------------------------------------

font = cv2.FONT_HERSHEY_SIMPLEX


cv2.putText(
    result,
    text1,
    (8, 22),
    font,
    0.45,
    (255, 255, 255),
    1,
    cv2.LINE_AA
)


cv2.putText(
    result,
    text2,
    (8, 45),
    font,
    0.45,
    (255, 255, 255),
    1,
    cv2.LINE_AA
)


cv2.putText(
    result,
    text3,
    (8, 68),
    font,
    0.45,
    (255, 255, 255),
    1,
    cv2.LINE_AA
)


cv2.putText(
    result,
    text4,
    (8, 94),
    font,
    0.45,
    (255, 255, 255),
    1,
    cv2.LINE_AA
)


# ------------------------------------------------------------
# 18. OUTPUT FILE NAME
# ------------------------------------------------------------

filename = os.path.splitext(
    os.path.basename(
        IMAGE_PATH
    )
)[0]


output_path = os.path.join(
    OUTPUT_DIR,
    filename + "_gradcam.jpg"
)


# ------------------------------------------------------------
# 19. SAVE RESULT
# ------------------------------------------------------------

success = cv2.imwrite(
    output_path,
    result
)


# ------------------------------------------------------------
# 20. FINISH
# ------------------------------------------------------------

print("\n============================================")

if success:

    print("GRAD-CAM COMPLETE")

    print("============================================")

    print("\nSaved result:")

    print(output_path)

else:

    print("ERROR: Could not save result.")

print("============================================")