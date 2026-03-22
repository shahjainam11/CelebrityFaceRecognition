r"""
retrain_model.py — Generate FaceNet embeddings and build an Annoy vector index.

This script scans the internal dataset, extracts robust facial embeddings using 
DeepFace (Facenet model), and builds an Approximate Nearest Neighbors (Annoy) 
index for lightning-fast matching, even with thousands of classes.

Run from the project root:
    .venv\Scripts\python.exe retrain_model.py
"""

import os
import sys
import json
from deepface import DeepFace
from annoy import AnnoyIndex

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(ROOT, "images_dataset")
ARTIFACTS_DIR = os.path.join(ROOT, "server", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".jfif", ".webp")
EMBEDDING_DIM = 128  # Facenet representation dimension
MODEL_NAME = "Facenet"

# ── Feature Extraction ───────────────────────────────────────────────────────
def get_embedding(img_path):
    """
    Extract the embedding for the main face in the image using DeepFace.
    Returns the vector, or None if no face is detected securely.
    """
    try:
        # DeepFace.represent returns a list of face representations
        res = DeepFace.represent(
            img_path=img_path, 
            model_name=MODEL_NAME, 
            enforce_detection=True,
            detector_backend="mtcnn"
        )
        if len(res) > 0:
            return res[0]["embedding"]
    except Exception as e:
        pass
    return None

# ── Build Dataset ─────────────────────────────────────────────────────────────
print("Scanning dataset for Deep Learning embeddings...")
classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])
print(f"Found classes: {classes}")

class_dict = {name: idx for idx, name in enumerate(classes)}

# Initialize Annoy Index
annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')

current_item_index: int = 0
item_to_class = {}  # Maps annoy_item_index -> class_label_integer

for cls in classes:
    cls_dir = os.path.join(DATASET_DIR, cls)
    count: int = 0
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith(EXTENSIONS):
            continue
        img_path = os.path.join(cls_dir, fname)
        
        emb = get_embedding(img_path)
        if emb is not None:
            annoy_index.add_item(current_item_index, emb)
            item_to_class[current_item_index] = class_dict[cls]
            current_item_index += 1
            count += 1
            
    print(f"  {cls}: {count} usable faces encoded")

print(f"\nTotal embedded samples: {current_item_index}")

if current_item_index == 0:
    print("ERROR: No faces could be processed. Check your dataset.")
    sys.exit(1)

# ── Build & Save Index ────────────────────────────────────────────────────────
print("\nBuilding Annoy Index (this is extremely fast)...")
annoy_index.build(50) 

model_path  = os.path.join(ARTIFACTS_DIR, "face_index.ann")
dict_path   = os.path.join(ARTIFACTS_DIR, "class_dictionary.json")
mapping_path = os.path.join(ARTIFACTS_DIR, "item_mapping.json")

annoy_index.save(model_path)

with open(dict_path, "w") as f:
    json.dump(class_dict, f)

with open(mapping_path, "w") as f:
    json.dump(item_to_class, f)

print(f"\nAnnoy Index saved to:   {model_path}")
print(f"Class Dict saved to:    {dict_path}")
print(f"Item Mapping saved to:  {mapping_path}")
print("Done! The model is now ready for Deep Learning inference.")
