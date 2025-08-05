from fastapi import FastAPI
from pydantic import BaseModel
from app.recommender import recommend_song

app = FastAPI(title="SwiftieMoodAgent")

class MoodIn(BaseModel):
    text: str

class RecOut(BaseModel):
    song_title: str
    similarity: float
    lyrics_excerpt: str

@app.post("/recommend", response_model=RecOut)
def recommend(mood: MoodIn):
    title, lyrics, score = recommend_song(mood.text)
    if title is None:
        return RecOut(song_title="N/A", similarity=0.0, lyrics_excerpt="")
    return RecOut(
        song_title=title,
        similarity=round(score, 3),
        lyrics_excerpt=(lyrics[:350] + "...") if lyrics else ""
    )