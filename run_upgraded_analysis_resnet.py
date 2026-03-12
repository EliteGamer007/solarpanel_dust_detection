import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import time

print("TensorFlow Version:", tf.__version__)
print("OpenCV Version:", cv2.__version__)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU is available and configured.")
else:
    print("No GPU found. Using CPU.")

DATASET_PATH = "Detect_solar_dust"
IMG_SIZE = (299, 299)

def build_inception_model():
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid", dtype="float32")(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(5e-5), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_resnet_model():
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid", dtype="float32")(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(5e-5), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------------------------------------
# Configuration for Multiple Models
# Team Members: Add your model details here!
# ---------------------------------------------------------
MODELS_CONFIG = {
    "inception": {
        "build_fn": build_inception_model,
        "preprocess_fn": __import__('tensorflow.keras.applications.inception_v3', fromlist=['preprocess_input']).preprocess_input,
        "img_size": (299, 299),
        "weights_path": "best_inception_solar.keras",
        "output_suffix": "inception"
    },
    
    # ---------------------------------------------------------
    # Member 1: ResNet50
    # ---------------------------------------------------------
    "resnet": {
        "build_fn": build_resnet_model,
        "preprocess_fn": tf.keras.applications.resnet50.preprocess_input,
        "img_size": (224, 224),
        "weights_path": "best_resnet_solar.keras",
        "output_suffix": "resnet"
    },
    
    # ---------------------------------------------------------
    # TODO: Member 2 (e.g., MobileNetV2)
    # ---------------------------------------------------------
    # "mobilenet": {
    #     "build_fn": build_mobilenet_model,
    #     "preprocess_fn": tf.keras.applications.mobilenet_v2.preprocess_input,
    #     "img_size": (224, 224),
    #     "weights_path": "best_mobilenet_solar.keras",
    #     "output_suffix": "mobilenet"
    # },
    
    # ---------------------------------------------------------
    # TODO: Member 3 (e.g., EfficientNetB0)
    # ---------------------------------------------------------
    # "efficientnet": {
    #     "build_fn": build_efficientnet_model,
    #     "preprocess_fn": tf.keras.applications.efficientnet.preprocess_input,
    #     "img_size": (224, 224),
    #     "weights_path": "best_efficientnet_solar.keras",
    #     "output_suffix": "efficientnet"
    # }
}

def predict_on_roi(roi_img, model, preprocess_fn, img_size):
    rgb_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_roi, img_size)
    input_tensor = preprocess_fn(tf.cast(np.expand_dims(resized, axis=0), tf.float32))
    pred = model.predict(input_tensor, verbose=0)[0][0]
    predicted_class = "Dusty" if pred > 0.5 else "Clean"
    confidence = pred if pred > 0.5 else 1.0 - pred
    return predicted_class, confidence

def locate_solar_panel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    
    if cv2.contourArea(largest_contour) < 5000: return None
        
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)

def process_video_headless(input_video_path, output_video_path, model, preprocess_fn, img_size):
    if not os.path.exists(input_video_path):
        print(f"Error: Video {input_video_path} not found.")
        return
        
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames from {input_video_path}...")
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        bbox = locate_solar_panel(frame)
        if bbox:
            x, y, w, h = bbox
            roi = frame[y:y+h, x:x+w]
            if w > 50 and h > 50:
                label, confidence = predict_on_roi(roi, model, preprocess_fn, img_size)
                color = (0, 0, 255)
                if label == "Clean": color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                text = f"{label} ({confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {frame_count}/{total_frames} frames... ({elapsed:.2f}s elapsed)")

    cap.release()
    out.release()
    print(f"\nVideo processing complete. Saved to {output_video_path}")

if __name__ == "__main__":
    input_vid = "Solar Panel Videos/Clean Solar Panel 1.mp4"
    
    print(f"Starting headless video rendering pipeline for all configured models...\n")
    
    for model_name, config in MODELS_CONFIG.items():
        print(f"--- Running Pipeline for Model: {model_name.upper()} ---")
        
        # 1. Build Model
        print(f"Building {model_name}...")
        model = config["build_fn"]()
        
        # 2. Load Weights 
        weights_path = config["weights_path"]
        if os.path.exists(weights_path):
            print(f"Loading existing weights from {weights_path}...")
            model.load_weights(weights_path)
        else:
            print(f"WARNING: No pre-trained weights found at '{weights_path}'. Model is untrained!")
            
        # 3. Process Video
        output_vid = f"Solar Panel Videos/clean_panel_1_processed_{config['output_suffix']}.mp4"
        process_video_headless(
            input_video_path=input_vid, 
            output_video_path=output_vid, 
            model=model, 
            preprocess_fn=config["preprocess_fn"], 
            img_size=config["img_size"]
        )
        print("-" * 50 + "\n")
        
    print("DONE. You can exit this session.")
