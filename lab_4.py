import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from google.colab import files
from google.colab.patches import cv2_imshow

# 1. LOAD THE PRE-TRAINED DETECTOR
MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
detector = hub.load(MODEL_URL)

# 2. COCO CLASS NAMES
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def detect_objects_with_boxes(image_path, detector, target_classes=None):
    """Run the detector on one image and return the image with boxes drawn.
    If target_classes is provided, only those classes will be labeled.
    """
    image = cv2.imread(image_path)
    orig_image = image.copy()

    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detector(input_tensor)

    num_detections = int(detections["num_detections"])
    detection_boxes = detections["detection_boxes"].numpy()[0]
    detection_classes = detections["detection_classes"].numpy()[0].astype(np.int32)
    detection_scores = detections["detection_scores"].numpy()[0]

    confidence_threshold = 0.5
    for i in range(num_detections):
        score = detection_scores[i]
        if score > confidence_threshold:
            class_id = detection_classes[i]
            class_name = COCO_CLASSES[class_id - 1] if 1 <= class_id <= 80 else "Unknown"
            
            # Filter to only draw specified target classes (e.g., ['car'])
            if target_classes is not None and class_name not in target_classes:
                continue
                
            box = detection_boxes[i]
            h, w, _ = orig_image.shape
            y_min, x_min, y_max, x_max = box
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            label = f"{class_name}: {score:.2f}"
            cv2.rectangle(orig_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(orig_image, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return orig_image

# 3. RUN IT - Filtered for 'car'
uploaded = files.upload()
if uploaded:
    image_path = next(iter(uploaded))
    # Set target_classes to ['car'] to only detect and outline cars
    result_image = detect_objects_with_boxes(image_path, detector, target_classes=["car"])
    cv2_imshow(result_image)


import os
import subprocess

try:
    import cv2
    ss_ready = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "segmentation")
except Exception:
    ss_ready = False

if not ss_ready:
    print("Installing opencv-contrib-python for Selective Search support...")
    subprocess.run("pip uninstall -y -q opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless", shell=True)
    subprocess.run("pip install -q opencv-contrib-python", shell=True)
    print("Restarting session to load new package. Please execute this cell again once restarted!")
    os.kill(os.getpid(), 9)

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# ---------------------------------------------------------------------------
# 2. CLASSIFIER: VGG16 pre-trained on ImageNet
# ---------------------------------------------------------------------------
model = VGG16(weights='imagenet', include_top=True)

# ---------------------------------------------------------------------------
# 3. REGION PROPOSALS: Selective Search (Fast), with a safe fallback
# ---------------------------------------------------------------------------
def get_region_proposals(img, max_regions=200):
    try:
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        boxes = [(x, y, w, h) for (x, y, w, h) in ss.process() if w > 40 and h > 40]
        if boxes:
            return boxes[:max_regions]
    except Exception:
        pass
    
    # fallback: multi-scale sliding windows
    H, W = img.shape[:2]
    boxes = []
    for s in (0.6, 0.4, 0.25):
        bw, bh = int(W*s), int(H*s)
        for y in range(0, H-bh+1, max(1, bh//2)):
            for x in range(0, W-bw+1, max(1, bw//2)):
                boxes.append((x, y, bw, bh))
    return boxes[:max_regions]

# ---------------------------------------------------------------------------
# 4. NON-MAXIMUM SUPPRESSION: drop duplicate overlapping boxes
# ---------------------------------------------------------------------------
def non_max_suppression(dets, iou_thr=0.3):
    if not dets:
        return []
    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h), _, _ in dets], dtype=float)
    scores = np.array([s for _, _, s in dets])
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0])
        yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2])
        yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        w_int = np.clip(xx2-xx1, 0, None)
        h_int = np.clip(yy2-yy1, 0, None)
        inter = w_int * h_int
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_r = (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1])
        order = rest[inter/(area_i+area_r-inter+1e-9) < iou_thr]
    return [dets[i] for i in keep]

# ---------------------------------------------------------------------------
# 5. RUN IT: upload, downscale, propose, classify (batched), draw
# ---------------------------------------------------------------------------
uploaded = files.upload()
if uploaded:
    image_path = next(iter(uploaded))
    img = cv2.imread(image_path)
    H0, W0 = img.shape[:2]
    img = cv2.resize(img, (600, int(H0 * 600.0 / W0)))

    boxes = get_region_proposals(img, max_regions=200)
    crops = np.array([cv2.resize(img[y:y+h, x:x+w], (224, 224)) for (x, y, w, h) in boxes], dtype=np.float32)
    preds = model.predict(preprocess_input(crops), batch_size=32, verbose=0)

    detections = []
    for (x, y, w, h), p in zip(boxes, preds):
        _, label, prob = decode_predictions(p[None], top=1)[0][0]
        if prob > 0.90:
            detections.append(((x, y, w, h), label, float(prob)))
            
    detections = non_max_suppression(detections, iou_thr=0.3)

    for (x, y, w, h), label, score in detections:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{label}: {score:.2f}", (x, max(18, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2_imshow(img)
    # ---------------------------------------------------------------------------
# DEBUGGING VERSION: Lower confidence threshold and print predictions
# ---------------------------------------------------------------------------
if 'img' in globals() and 'boxes' in globals() and 'preds' in globals():
    print(f"[Debug] Selective Search proposed {len(boxes)} regions.")
    
    # Make a copy of the image for debugging
    debug_img = img.copy()
    
    detections = []
    print("\n[Debug] Sample predictions with confidence > 50%:")
    for idx, ((x, y, w, h), p) in enumerate(zip(boxes, preds)):
        _, label, prob = decode_predictions(p[None], top=1)[0][0]
        prob = float(prob)
        
        # Lowered threshold to 50% to see what the model is predicting
        if prob > 0.50:
            if len(detections) < 10:
                print(f"Region {idx}: Predicted '{label}' with {prob*100:.1f}% confidence")
            detections.append(((x, y, w, h), label, prob))
            
    # Apply Non-Maximum Suppression to remove duplicates
    filtered_detections = non_max_suppression(detections, iou_thr=0.3)
    print(f"\n[Debug] Found {len(filtered_detections)} objects after filtering (IoU threshold 0.3).")
    
    # Draw the boxes
    for (x, y, w, h), label, score in filtered_detections:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(debug_img, f"{label}: {score:.2f}", (x, max(18, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2_imshow(debug_img)
else:
    print("Please run the setup/upload cell (0e3dee64) first before running this debugging cell!")