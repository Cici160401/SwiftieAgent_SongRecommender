from recommender import get_random_lyrics_sample

sample = get_random_lyrics_sample(5)
for title, lyrics in sample.items():
    print(f"\n🎵 {title}:\n{lyrics[:300]}...\n")