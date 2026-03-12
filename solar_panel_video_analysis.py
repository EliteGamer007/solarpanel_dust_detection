import os
import cv2
import numpy as np
import tensorflow as tf
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

# ---------------------------------------------------------
# Model Building Functions
# ---------------------------------------------------------

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

def build_vgg_model():
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid", dtype="float32")(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(5e-5), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_densenet_model():
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
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
# Configuration for All Best Models
# ---------------------------------------------------------

MODELS_CONFIG = {
    "inception": {
        "build_fn": build_inception_model,
        "preprocess_fn": __import__('tensorflow.keras.applications.inception_v3', fromlist=['preprocess_input']).preprocess_input,
        "img_size": (299, 299),
        "weights_path": "best_inception_solar.keras",
        "output_suffix": "inception"
    },
    "resnet": {
        "build_fn": build_resnet_model,
        "preprocess_fn": __import__('tensorflow.keras.applications.resnet50', fromlist=['preprocess_input']).preprocess_input,
        "img_size": (224, 224),
        "weights_path": "best_resnet_solar.keras",
        "output_suffix": "resnet"
    },
    "vgg": {
        "build_fn": build_vgg_model,
        "preprocess_fn": __import__('tensorflow.keras.applications.vgg16', fromlist=['preprocess_input']).preprocess_input,
        "img_size": (224, 224),
        "weights_path": "best_vgg_solar.keras",
        "output_suffix": "vgg"
    },
    "densenet": {
        "build_fn": build_densenet_model,
        "preprocess_fn": __import__('tensorflow.keras.applications.densenet', fromlist=['preprocess_input']).preprocess_input,
        "img_size": (224, 224),
        "weights_path": "best_densenet_solar.keras",
        "output_suffix": "densenet"
    }
}

# ---------------------------------------------------------
# Video Processing Core Logic
# ---------------------------------------------------------

def predict_on_roi(roi_img, model, preprocess_fn, img_size):
    """Prepares ROI for prediction."""
    rgb_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_roi, img_size)
    input_tensor = preprocess_fn(tf.cast(np.expand_dims(resized, axis=0), tf.float32))
    pred = model.predict(input_tensor, verbose=0)[0][0]
    predicted_class = "Dusty" if pred > 0.5 else "Clean"
    confidence = pred if pred > 0.5 else 1.0 - pred
    return predicted_class, confidence

def locate_solar_panel(frame):
    """Contours-based panel detection."""
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
    """Processes video and saves output with tracking overlay."""
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
                color = (0, 0, 255) # Red for Dusty
                if label == "Clean": color = (0, 255, 0) # Green for Clean

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                text = f"{label} ({confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                # Position background rectangle and text INSIDE the bounding box (top-left)
                cv2.rectangle(frame, (x, y), (x + tw + 10, y + th + 10), color, -1)
                cv2.putText(frame, text, (x + 5, y + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {frame_count}/{total_frames} frames... ({elapsed:.2f}s elapsed)")

    cap.release()
    out.release()
    print(f"\nSaved result to: {output_video_path}")

# ---------------------------------------------------------
# Main Execution Hub
# ---------------------------------------------------------

if __name__ == "__main__":
    # Target Video
    input_video = "Solar Panel Videos/Clean Solar Panel 1.mp4"
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    
    print(f"\n{'='*50}")
    print(f"SOLAR PANEL VIDEO ANALYSIS SYSTEM")
    print(f"{'='*50}\n")
    
    for model_name, config in MODELS_CONFIG.items():
        print(f"\n>>> Running Pipeline for: {model_name.upper()}")
        
        weights_path = config["weights_path"]
        
        # 1. Load Model (Architecture + Weights)
        if os.path.exists(weights_path):
            print(f"Loading best weights from {weights_path}...")
            try:
                # Preferred: Load full model if saved as such
                model = tf.keras.models.load_model(weights_path)
            except Exception:
                # Fallback: Build and load weights manually
                print(f"Restoring architecture and loading weights...")
                model = config["build_fn"]()
                model.load_weights(weights_path)
        else:
            print(f"WARNING: No weight file found at '{weights_path}'. Skipping.")
            continue
            
        # 2. Process Target Video
        # Dynamic naming: <video_name>_<model_suffix>.mp4
        output_filename = f"Solar Panel Videos/{video_name}_{config['output_suffix']}.mp4"
        process_video_headless(
            input_video_path=input_video, 
            output_video_path=output_filename, 
            model=model, 
            preprocess_fn=config["preprocess_fn"], 
            img_size=config["img_size"]
        )
        print("-" * 50)
        
    print("\nALL MODELS PROCESSED. COMPARISON VIDEOS READY.")
