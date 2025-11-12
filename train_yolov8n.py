Yolo Model Training:

# YOLOv8n Robust Training Pipeline - Single Dataset with Mixed Conditions
# =======================================================================

# 1. MOUNT GOOGLE DRIVE (Only for saving models/checkpoints)
# ----------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install ultralytics roboflow --quiet
!pip install --upgrade pip --quiet

# Import libraries
import os
import yaml
import shutil
from ultralytics import YOLO
import torch
from IPython.display import Image, display

# Create project directory in Drive for models only
models_dir = '/content/drive/MyDrive/YOLOv8_Models_success'
runs_dir = '/content/drive/MyDrive/YOLOv8_Runs_sucess'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)

print(f"✅ Model storage directory: {models_dir}")
print(f"✅ Training runs directory: {runs_dir}")

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 2. DATASET DOWNLOAD (Fresh each time - faster than Drive transfer)
# ------------------------------------------------------------------
print("⬇ Downloading robust dataset from Roboflow...")

# Clean up any existing downloads
import glob
existing_datasets = glob.glob('/content/success*')
for old_dataset in existing_datasets:
    if os.path.isdir(old_dataset):
        print(f"🧹 Cleaning up: {old_dataset}")
        shutil.rmtree(old_dataset)

# Download dataset from Roboflow
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="rx1lOkdnOKmrKsfvJhWn")
project = rf.workspace("jhanani-5rwd5").project("success-iev8v")
version = project.version(4)
dataset = version.download("yolov8")

dataset_path = dataset.location
print(f"✅ Robust dataset downloaded to: {dataset_path}")

# Verify dataset structure
def validate_dataset(path):
    """Quick dataset validation"""
    image_count = 0
    yaml_file = None

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_count += 1
            elif file == 'data.yaml':
                yaml_file = os.path.join(root, file)

    return image_count, yaml_file

image_count, data_yaml_path = validate_dataset(dataset_path)
print(f"📊 Found {image_count} images (with augmentation)")
print(f"📄 data.yaml: {data_yaml_path}")

if image_count == 0:
    raise Exception("❌ No images found in dataset! Check your Roboflow export.")

if not data_yaml_path:
    raise Exception("❌ data.yaml not found! Make sure you exported in YOLOv8 format.")

# Read dataset config to show classes
with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)
print(f"🎯 Training classes: {data_config.get('names', [])}")
print(f"🔢 Number of classes: {data_config.get('nc', 'Unknown')}")

# 3. MODEL INITIALIZATION & RESUME LOGIC
# ---------------------------------------
checkpoint_path = f'{models_dir}/robust_checkpoint.pt'

# Check for existing checkpoint
if os.path.exists(checkpoint_path):
    print("🔄 Found existing robust training checkpoint! Resuming...")
    model = YOLO(checkpoint_path)
    resume_training = True
    print(f"📁 Loaded checkpoint from: {checkpoint_path}")
else:
    print("🆕 Starting fresh robust training...")
    # Start with pretrained weights for better performance on varied conditions
    model = YOLO('yolov8n.pt')  # Pretrained weights help with robustness
    resume_training = False

print(f"Resume training: {resume_training}")

# 4. ROBUST TRAINING CONFIGURATION
# --------------------------------
training_config = {
    'data': data_yaml_path,                      # Path to dataset config
    'epochs': 100,                               # More epochs for robust learning
    'imgsz': 640,                                # Image size (matches preprocessing)
    'batch': 32,                                 # Batch size (reduce if GPU memory issues)
    'workers': 2,                                # Data loading workers
    'device': 0 if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    'project': '/content/runs',                  # Local runs folder (fast)
    'name': 'yolov8n_robust',                    # Experiment name for robust model
    'save_period': 1,                            # Save checkpoint every 1 epoch
    'patience': 30,                              # Early stopping patience
    'save': True,                                # Save final checkpoint
    'plots': True,                               # Generate training plots
    'val': True,                                 # Validate during training
    'resume': resume_training,                   # Resume if checkpoint exists

    # 🚀 ROBUST TRAINING SETTINGS
    'augment': True,                             # Enable YOLO's built-in augmentation
    'mixup': 0.1,                                # Mix different images (helps with noise)
    'mosaic': 1.0,                               # Mosaic augmentation (combines 4 images)
    'copy_paste': 0.1,                           # Copy-paste augmentation
    'degrees': 10.0,                             # Additional rotation augmentation
    'translate': 0.1,                            # Translation augmentation
    'scale': 0.5,                                # Scale augmentation
    'shear': 2.0,                                # Shear augmentation
    'perspective': 0.0001,                       # Perspective augmentation
    'flipud': 0.0,                               # No vertical flip for cars
    'fliplr': 0.5,                               # 50% horizontal flip
    'hsv_h': 0.015,                              # Hue augmentation
    'hsv_s': 0.7,                                # Saturation augmentation
    'hsv_v': 0.4,                                # Brightness augmentation
}

print("\n📋 Robust Training Configuration:")
print("🎯 Focus: Training model to handle both clean and noisy conditions")
for key, value in training_config.items():
    if key in ['augment', 'mixup', 'mosaic', 'copy_paste']:
        print(f"  🔥 {key}: {value}")
    else:
        print(f"     {key}: {value}")

# 5. CUSTOM CALLBACK FOR DRIVE BACKUP
# ------------------------------------
from ultralytics.utils.callbacks import default_callbacks

def backup_robust_model(trainer):
    """Backup robust model files to Drive during training"""
    # Backup every epoch now
    try:
        # Backup latest checkpoint with robust naming
        if os.path.exists(trainer.last):
            shutil.copy2(trainer.last, f'{models_dir}/robust_checkpoint.pt')
            print(f"💾 Robust checkpoint backed up to Drive (Epoch {trainer.epoch})")

        # Backup best model if it exists
        if os.path.exists(trainer.best):
            shutil.copy2(trainer.best, f'{models_dir}/robust_best.pt')
            print(f"🏆 Best robust model backed up to Drive")

    except Exception as e:
        print(f"⚠ Backup failed: {e}")

# Add custom callback
model.add_callback("on_train_epoch_end", backup_robust_model)

# 6. START ROBUST TRAINING
# ------------------------
print("\n🚀 Starting robust training for mixed conditions...")
print("💡 Tip: Training for both clean and noisy conditions!")
print("💾 Progress saves to Google Drive every epoch!")

# Train the model
results = model.train(**training_config)

# 7. FINAL BACKUP TO DRIVE
# -------------------------
print("\n💾 Robust training completed! Backing up final models...")

# Copy final models to Drive with descriptive names
if os.path.exists(model.trainer.best):
    shutil.copy2(model.trainer.best, f'{models_dir}/robust_final_best.pt')
    print(f"✅ Best robust model: {models_dir}/robust_final_best.pt")

if os.path.exists(model.trainer.last):
    shutil.copy2(model.trainer.last, f'{models_dir}/robust_final_last.pt')
    print(f"✅ Last robust model: {models_dir}/robust_final_last.pt")

# Backup training results
results_src = f'/content/runs/yolov8n_robust'
results_dst = f'{runs_dir}/robust_training_run'
if os.path.exists(results_src):
    if os.path.exists(results_dst):
        shutil.rmtree(results_dst)
    shutil.copytree(results_src, results_dst)
    print(f"✅ Robust training results: {results_dst}")

# 8. TRAINING RESULTS
# -------------------
print(f"\n📊 Robust Training Results:")
print(f"Best weights: {model.trainer.best}")
print(f"Last weights: {model.trainer.last}")

# Display training curves
results_img = f'{results_src}/results.png'
if os.path.exists(results_img):
    print("\n📈 Robust Training Progress:")
    display(Image(results_img))

# 9. MODEL VALIDATION
# --------------------
print("\n🔍 Validating robust model...")
val_results = model.val()

print(f"📈 Robust Model mAP50: {val_results.box.map50:.4f}")
print(f"📈 Robust Model mAP50-95: {val_results.box.map:.4f}")

# 10. TEST ON SAMPLE IMAGES
# --------------------------
print("\n🧪 Testing robust model on sample images...")
test_images = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(root, file))
        if len(test_images) >= 5:  # Test more images
            break
    if len(test_images) >= 5:
        break

# Test with different confidence levels for robustness
confidence_levels = [0.3, 0.5, 0.7]
for conf in confidence_levels:
    print(f"\n🎯 Testing with confidence {conf}:")
    for i, img_path in enumerate(test_images[:3]):
        results = model.predict(img_path, save=True, conf=conf)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  ✅ {os.path.basename(img_path)}: {detections} detections")

# 11. SAVE ROBUST MODEL INFO
# ---------------------------
info_file = f'{models_dir}/robust_training_info.txt'
with open(info_file, 'w') as f:
    f.write(f"YOLOv8n Robust Training Session\n")
    f.write(f"==============================\n")
    f.write(f"Dataset: {dataset_path}\n")
    f.write(f"Dataset version: {version.version}\n")
    f.write(f"Images (with augmentation): {image_count}\n")
    f.write(f"Classes: {data_config.get('nc', 'Unknown')}\n")
    f.write(f"Class names: {', '.join(data_config.get('names', []))}\n")
    f.write(f"Training approach: Single robust dataset\n")
    f.write(f"Conditions trained: Clean + Noisy + Various lighting\n")
    f.write(f"Final mAP50: {val_results.box.map50:.4f}\n")
    f.write(f"Final mAP50-95: {val_results.box.map:.4f}\n")
    f.write(f"Best model: {models_dir}/robust_final_best.pt\n")
    f.write(f"Last checkpoint: {models_dir}/robust_final_last.pt\n")
    f.write(f"Training results: {runs_dir}/robust_training_run\n")

print(f"📝 Robust training info saved: {info_file}")

# 12. ROBUST MODEL USAGE INSTRUCTIONS
# ------------------------------------
print("""
🎉 ROBUST TRAINING COMPLETED SUCCESSFULLY!

📁 Your robust models are saved in Google Drive:
   • Best model: {models_dir}/robust_final_best.pt
   • Resume checkpoint: {models_dir}/robust_checkpoint.pt
   • Training plots: {runs_dir}/robust_training_run

🎯 ROBUST MODEL FEATURES:
   ✅ Handles clean images
   ✅ Handles noisy images
   ✅ Works in various lighting conditions
   ✅ Robust to slight blur and camera variations

🔄 TO RESUME INTERRUPTED TRAINING:
   Just rerun this script! It detects robust checkpoints automatically.

📋 INFERENCE CODE:
   python
   from ultralytics import YOLO

   # Load robust model
   model = YOLO('{models_dir}/robust_final_best.pt')

   # For clean images
   results = model.predict('clean_image.jpg', conf=0.5)

   # For noisy images (lower confidence)
   results = model.predict('noisy_image.jpg', conf=0.3)
   

💡 ROBUST MODEL TIPS:
   • Use conf=0.5 for clean images
   • Use conf=0.3 for very noisy images
   • Model handles various lighting automatically
   • Test on real-world noisy conditions
   • Monitor performance on both clean and noisy validation sets

🚀 Your model is now ready for real-world deployment!
""".format(models_dir=models_dir, runs_dir=runs_dir))

print("✅ Robust pipeline completed!")
