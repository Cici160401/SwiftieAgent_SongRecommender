This is a MiniLM and zero-shot embeddings api to recommend songs depending on your mood.


texto usuario ──► encode (SBERT) ─►  u (1×768)
                                ▲
        matriz X (208×768)  ◄───┘   similarities = X·u  → argmax → idx
               ▲                               │
     embeddings precalculados                  │
               ▲                               ▼
   build_index.py   ↲  (npz + json)   if sim<0.15 → "ninguna"
                                else → título + letra + score