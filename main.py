import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
from typing import List

from semaforce.chunker import SemanticChunker


def main():
    with open("data.txt", "r") as file:
        text = file.read()

    chunker = SemanticChunker(max_tokens=100, window_size=7)  # Setting a low token limit for demonstration
    chunks = chunker.chunk_text(text)

    print("Semantic Chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} (Tokens: {len(chunker.tokenizer.encode(chunk))}):")
        print(chunk)

if __name__ == "__main__":
    main()