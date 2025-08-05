import re, requests, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------
# 1. Rutas absolutas
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # swiftie-agent/
DATA_DIR = BASE_DIR / "data"
EMB_PATH = DATA_DIR / "embeddings.npz"
META_PATH = DATA_DIR / "meta.json"

# -------------------------------------------------------------
# 2. Carga del modelo — ahora all-mpnet-base-v2
# -------------------------------------------------------------
#Descarga y mantiene en memoria un encoder de 768 dimensiones entrenado para capturar significado y tono 
#Model:  sentence-transformers/all-mpnet-base-v2 
model = SentenceTransformer("all-mpnet-base-v2")           

# -------------------------------------------------------------
# 3. Carga de embeddings y metadatos
# -------------------------------------------------------------
if EMB_PATH.exists() and META_PATH.exists():
    #X contiene un vector por canción, ya normalizado (o se normaliza al vuelo si detectamos que no tiene norma 1).
    X = np.load(EMB_PATH)["X"].astype("float32")           # (N, 768)

    #se deben normalizar si no lo están
    if abs(np.linalg.norm(X[0]) - 1.0) > 1e-3:
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    #meta es una lista paralela de dicts
    meta = json.load(META_PATH.open(encoding="utf-8"))

    print(f" Embeddings cargados: {X.shape[0]} canciones – dim {X.shape[1]}")
else:
    print(f" Embeddings NO encontrados en {EMB_PATH}")
    X, meta = None, []

# -------------------------------------------------------------
# 4. Helpers de limpieza y API externa
# -------------------------------------------------------------
BASE_URL = "https://taylor-swift-api.sarbo.workers.dev"

def get_all_songs():
    #Here we are getting the complete list of songs from Taylor's api
    try:
        response = requests.get(f"{BASE_URL}/songs")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print("Error al obtener la lista de canciones:",e)
        return []


def clean_lyrics(txt: str) -> str:
    #Quita tags [Chorus]… y devuelve la letra completa sin adornos.
    #Es decir dejamos la letra completa solamente para que el embedding capte el tono
    return re.sub(r"\[.*?]", "", txt).strip()


def get_lyrics(song_id: int) -> str:
    #Hace el llamado a la API de taylor para obtener las letras
    try:
        r = requests.get(f"{BASE_URL}/lyrics/{song_id}")
        r.raise_for_status()
        return r.json().get("lyrics", "")
    except requests.RequestException:
        return ""

# -------------------------------------------------------------
# 5. Motor de recomendación
# -------------------------------------------------------------
#

def recommend_song(user_text: str, min_sim: float = 0.15):
    """
    Devuelve (titulo, letra, similitud).  Lanza RuntimeError si aún no hay índice.
    """
    if X is None:
        raise RuntimeError("Embeddings no cargados; ejecuta build_index.py primero.")
    if not user_text.strip():
        return "Texto vacío", "", 0.0

    u = model.encode(user_text, normalize_embeddings=True)     # (768,)
    sims = X @ u                                               # coseno

    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])

    #Umbral de confianza, evita recomendar algo si la similitud a las letras es demasiado baja, es decir más baja que 0.15
    if best_sim < min_sim:
        return "No song seems to be similar to your emotion, try again with another one 😕", "", best_sim
    #Escogemos la canción ganadora
    best_meta = meta[best_idx]
    #Se devuelve un fragmento de la canción con similitud
    excerpt   = clean_lyrics(get_lyrics(best_meta["song_id"]))[:250] + "…"

    return best_meta["title"], excerpt, round(best_sim, 3)