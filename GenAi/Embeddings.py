from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# Sample data (typically you'd train this on a larger corpus)
sentences = [
    ['dog', 'barks'],
    ['cat', 'meows'],
    ['dog', 'runs'],
    ['cat', 'sleeps'],
    ['dog', 'chases', 'cat']
]

# Train a Word2Vec model on the sample data
model = Word2Vec(sentences, min_count=1)

# Get embeddings for specific words
dog_embedding = model.wv['dog']
cat_embedding = model.wv['cat']

# Calculate similarity between words (cosine similarity)
similarity = model.wv.similarity('dog', 'cat')

print(f"Embedding for 'dog': {dog_embedding}")
print(f"Embedding for 'cat': {cat_embedding}")
print(f"Similarity between 'dog' and 'cat': {similarity}")
