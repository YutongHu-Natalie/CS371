import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

revision= None
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0", revision=revision
                            )
model = model.to(device)

texts= ["hello, world", "Yes, I am", "I know you"]

# Compute embeddings
embeddings = model.encode(
    texts,
    convert_to_tensor=True,
    device=device,
    batch_size=32,
    show_progress_bar=True
)
# Compute cosine-similarity for each pair of sentences
scores = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

print(scores.cpu().numpy())