import torch
import torch.nn.functional as F
from conda.exports import root_dir
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pprint
import dotenv

# Get the list of user's
load_dotenv()

CLAUDE_KEY = os.getenv('Claude_key')
GPT_KEY = os.getenv('gpt_key')
print(CLAUDE_KEY, GPT_KEY)
