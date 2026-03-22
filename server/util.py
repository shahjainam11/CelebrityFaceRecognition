import json
import numpy as np
import base64
import cv2
import os
from PIL import Image
from deepface import DeepFace
from annoy import AnnoyIndex

# ─── Globals ──────────────────────────────────────────────────────────────────
__class_name_to_number = {}
__class_number_to_name = {}
__item_mapping = {}

EMBEDDING_DIM = 128
MODEL_NAME = "Facenet"

__annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')

# ─── Celebrity Info ────────────────────────────────────────────────────────────
SPORT_MAP = {
    "lionel_messi": "Football",
    "maria_sharapova": "Tennis",
    "roger_federer": "Tennis",
    "serena_williams": "Tennis",
    "virat_kohli": "Cricket",
    "ms_dhoni": "Cricket",
    "rohit_sharma": "Cricket",
    "hardik_pandya": "Cricket",
    "k._l._rahul": "Cricket",
    "jasprit_bumrah": "Cricket",
    "ravindra_jadeja": "Cricket",
    "bhuvneshwar_kumar": "Cricket",
    "shikhar_dhawan": "Cricket",
    "dinesh_karthik": "Cricket",
    "kedar_jadhav": "Cricket",
    "kuldeep_yadav": "Cricket",
    "mohammed_shami": "Cricket",
    "yuzvendra_chahal": "Cricket",
    "vijay_shankar": "Cricket"
}

SPORT_EMOJI_MAP = {
    "Football": "⚽",
    "Tennis": "🎾",
    "Cricket": "🏏",
}

CELEBRITY_INFO = {
    "lionel_messi": {"display_name": "Lionel Messi", "sport": "Football", "nationality": "Argentina", "born": "June 24, 1987", "titles": "8x Ballon d'Or", "club": "Inter Miami CF", "fun_fact": "Scored 91 goals in a single calendar year (2012)!", "sport_emoji": "⚽"},
    "maria_sharapova": {"display_name": "Maria Sharapova", "sport": "Tennis", "nationality": "Russia", "born": "April 19, 1987", "titles": "5x Grand Slam", "club": "Retired", "fun_fact": "She moved to the US at age 7 with just $700.", "sport_emoji": "🎾"},
    "roger_federer": {"display_name": "Roger Federer", "sport": "Tennis", "nationality": "Switzerland", "born": "August 8, 1981", "titles": "20x Grand Slam", "club": "Retired", "fun_fact": "He held the world No. 1 spot for 237 consecutive weeks.", "sport_emoji": "🎾"},
    "serena_williams": {"display_name": "Serena Williams", "sport": "Tennis", "nationality": "USA", "born": "September 26, 1981", "titles": "23x Grand Slam", "club": "Retired", "fun_fact": "She won the 2017 Australian Open while 8 weeks pregnant.", "sport_emoji": "🎾"},
    "virat_kohli": {"display_name": "Virat Kohli", "sport": "Cricket", "nationality": "India", "born": "November 5, 1988", "titles": "ODI WC, T20 WC", "club": "RCB", "fun_fact": "He holds the record for most hundreds in ODI cricket.", "sport_emoji": "🏏"},
    "ms_dhoni": {"display_name": "MS Dhoni", "sport": "Cricket", "nationality": "India", "born": "July 7, 1981", "titles": "T20 WC, ODI WC, CT", "club": "CSK", "fun_fact": "The only captain to win all ICC major trophies.", "sport_emoji": "🏏"},
    "rohit_sharma": {"display_name": "Rohit Sharma", "sport": "Cricket", "nationality": "India", "born": "April 30, 1987", "titles": "T20 WC (x2)", "club": "Mumbai Indians", "fun_fact": "Holds the record for the highest individual score in ODIs (264).", "sport_emoji": "🏏"},
    "hardik_pandya": {"display_name": "Hardik Pandya", "sport": "Cricket", "nationality": "India", "born": "October 11, 1993", "titles": "T20 WC", "club": "Mumbai Indians", "fun_fact": "One of India's most destructive fast-bowling all-rounders.", "sport_emoji": "🏏"},
    "k._l._rahul": {"display_name": "KL Rahul", "sport": "Cricket", "nationality": "India", "born": "April 18, 1992", "titles": "Asia Cup", "club": "LSG", "fun_fact": "Known for his elegant batting style across all formats.", "sport_emoji": "🏏"},
    "jasprit_bumrah": {"display_name": "Jasprit Bumrah", "sport": "Cricket", "nationality": "India", "born": "December 6, 1993", "titles": "T20 WC", "club": "Mumbai Indians", "fun_fact": "Famous for his unique slinging action and brutal yorkers.", "sport_emoji": "🏏"},
    "ravindra_jadeja": {"display_name": "Ravindra Jadeja", "sport": "Cricket", "nationality": "India", "born": "December 6, 1988", "titles": "CT, T20 WC", "club": "CSK", "fun_fact": "Nicknamed 'Sir Jadeja', known for his incredible fielding.", "sport_emoji": "🏏"},
    "bhuvneshwar_kumar": {"display_name": "Bhuvneshwar Kumar", "sport": "Cricket", "nationality": "India", "born": "February 5, 1990", "titles": "CT", "club": "SRH", "fun_fact": "One of the best swing bowlers in modern Indian cricket.", "sport_emoji": "🏏"},
    "shikhar_dhawan": {"display_name": "Shikhar Dhawan", "sport": "Cricket", "nationality": "India", "born": "December 5, 1985", "titles": "CT", "club": "PBKS", "fun_fact": "Known as 'Gabbar' for his iconic mustache twirl.", "sport_emoji": "🏏"},
    "dinesh_karthik": {"display_name": "Dinesh Karthik", "sport": "Cricket", "nationality": "India", "born": "June 1, 1985", "titles": "T20 WC, CT", "club": "RCB", "fun_fact": "Made a legendary comeback as a finisher late in his career.", "sport_emoji": "🏏"},
    "kedar_jadhav": {"display_name": "Kedar Jadhav", "sport": "Cricket", "nationality": "India", "born": "March 26, 1985", "titles": "Asia Cup", "club": "Retired", "fun_fact": "Famous for his extremely low-arm bowling action.", "sport_emoji": "🏏"},
    "kuldeep_yadav": {"display_name": "Kuldeep Yadav", "sport": "Cricket", "nationality": "India", "born": "December 14, 1994", "titles": "T20 WC", "club": "DC", "fun_fact": "A rare left-arm unorthodox spin bowler (chinaman).", "sport_emoji": "🏏"},
    "mohammed_shami": {"display_name": "Mohammed Shami", "sport": "Cricket", "nationality": "India", "born": "September 3, 1990", "titles": "Asia Cup", "club": "GT", "fun_fact": "Known for having one of the best seam presentations in the world.", "sport_emoji": "🏏"},
    "yuzvendra_chahal": {"display_name": "Yuzvendra Chahal", "sport": "Cricket", "nationality": "India", "born": "July 23, 1990", "titles": "T20 WC", "club": "RR", "fun_fact": "He is a former chess player who represented India in youth tournaments.", "sport_emoji": "🏏"},
    "vijay_shankar": {"display_name": "Vijay Shankar", "sport": "Cricket", "nationality": "India", "born": "January 26, 1991", "titles": "Asia Cup", "club": "GT", "fun_fact": "A highly rated all-rounder with excellent technique.", "sport_emoji": "🏏"},
}

# ─── Artifact loading ─────────────────────────────────────────────────────────
def load_saved_artifacts():
    global __class_name_to_number, __class_number_to_name, __item_mapping, __annoy_index
    print("Loading Deep Learning artifacts...")
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    
    with open(os.path.join(artifacts_dir, "class_dictionary.json"), "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
        
    with open(os.path.join(artifacts_dir, "item_mapping.json"), "r") as f:
        # JSON keys are strings, convert back to int
        __item_mapping = {int(k): v for k, v in json.load(f).items()}
        
    __annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')
    __annoy_index.load(os.path.join(artifacts_dir, "face_index.ann"))
    print("Annoy Index loaded successfully.")

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def get_cv2_image_from_pil(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def classify_image(image_base64_data, file_path=None):
    """Legacy API for backward compat."""
    return []

# ─── Prediction ───────────────────────────────────────────────────────────────
def two_step_predict(pil_image):
    """
    Step 1 — DeepFace Feature Extraction
    Step 2 — Annoy Vector Matching
    """
    img_bgr = get_cv2_image_from_pil(pil_image)
    
    try:
        # represent returns [{embedding, facial_area, face_confidence}]
        res = DeepFace.represent(
            img_path=img_bgr, 
            model_name=MODEL_NAME, 
            enforce_detection=True, 
            detector_backend="mtcnn"
        )
        if len(res) == 0:
            return None
            
        face_data = res[0]
        emb = face_data["embedding"]
        facial_area = face_data["facial_area"]
        rect = (facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"])
        
    except Exception as e:
        return None
        
    # Query Annoy Index for top neighbors
    nns, distances = __annoy_index.get_nns_by_vector(emb, 10, include_distances=True)
    
    if len(nns) == 0:
        return None
        
    # The best match
    best_item = nns[0]
    best_dist = distances[0]
    
    # Calculate crude confidence from angular distance (0 = identical, 1.41 = orthogonal)
    confidence = max(0.0, (1.0 - (best_dist**2) / 2.0)) * 100.0
    
    pred_num = __item_mapping[best_item]
    player_name = __class_number_to_name[pred_num]

    # Reject poor matches instead of confidently guessing the wrong person
    if confidence < 50.0:
        player_name = "unknown_match"
        sport = "Unknown"
        sport_emoji = "❓"
    else:
        sport = SPORT_MAP.get(player_name, "Unknown Sport")
        sport_emoji = SPORT_EMOJI_MAP.get(sport, "🏅")
    
    # Probabilities for other classes (crude mapping for UI)
    prob_dict = {}
    for cl_name in __class_name_to_number.keys():
        prob_dict[cl_name] = 0.0
        
    # distribute weights based on distances
    for n_id, d_val in zip(nns, distances):
        sim = max(0.0, (1.0 - (d_val**2)/2.0)) * 100.0
        label_num = __item_mapping[n_id]
        label_name = __class_number_to_name[label_num]
        prob_dict[label_name] = max(prob_dict[label_name], sim)
        
    prob_dict_sorted = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    info = CELEBRITY_INFO.get(player_name, {})

    return {
        "player_key": player_name,
        "player_display": info.get("display_name", player_name.replace("_", " ").title()),
        "confidence": round(confidence, 2),
        "sport": sport,
        "sport_emoji": sport_emoji,
        "probabilities": prob_dict_sorted,
        "class_dictionary": __class_name_to_number,
        "face_rect": rect,
        "info": info,
    }

if __name__ == "__main__":
    load_saved_artifacts()
