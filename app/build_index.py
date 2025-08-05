import os, json, time
from pathlib import Path,PurePath
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from recommender import get_all_songs, get_lyrics, clean_lyrics   # reutilizamos helpers

# ----------------- Parámetros -----------------
MODEL_NAME = "all-mpnet-base-v2"
PAUSE_SEC  = 1.2                      # pausa entre peticiones → evita 429
BASE_DIR = Path(__file__).resolve().parent.parent   # swiftie-agent/
OUT_DIR  = BASE_DIR / "data"                       # 👈  cambia esto
OUT_DIR.mkdir(exist_ok=True)
# ----------------------------------------------

def main() -> None:
    print("Descargando lista de canciones…")
    songs = get_all_songs()
    print(f"Total canciones: {len(songs)}")

    model = SentenceTransformer(MODEL_NAME)
    dim   = model.get_sentence_embedding_dimension()    # 768 para MPNet
    print(f"Modelo: {MODEL_NAME}  ·  dimensión: {dim}")

    vectors = []
    meta    = []

    for s in tqdm(songs, desc="Embed", unit="canción"):
        txt_raw = get_lyrics(s["song_id"])
        txt     = clean_lyrics(txt_raw)
        if not txt:                           # letra vacía o error
            time.sleep(PAUSE_SEC)
            continue

        vec = model.encode(txt, normalize_embeddings=True)   # (dim,)
        vectors.append(vec)
        meta.append({"song_id": s["song_id"], "title": s["title"]})

        time.sleep(PAUSE_SEC)                # evita rate-limit 429

    # ---------- Guardar ----------
    X = np.vstack(vectors).astype("float32")                # (N,dim)
    np.savez_compressed(OUT_DIR / "embeddings.npz", X=X)

    with (OUT_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nGuardados {len(meta)} embeddings ✓")
    print("Archivos creados en", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
