import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
from typing import List

class SemanticChunker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', window_size: int = 3,
                 threshold: float = 0.2, max_tokens: int = 8192, model: str = "gpt-4o"):
        self.sentence_model = SentenceTransformer(model_name, cache_folder='.cache')
        self.window_size = window_size
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.encoding_for_model(model)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_embedding(self, text: str) -> np.ndarray:
        return self.sentence_model.encode([text], show_progress_bar=False)[0]

    def split_into_sentences(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        sentences = []
        current_sentence = []
        for token in tokens:
            current_sentence.append(token)
            if token in [13, 30]:  # Common end-of-sentence tokens
                sentences.append(self.tokenizer.decode(current_sentence))
                current_sentence = []
        if current_sentence:
            sentences.append(self.tokenizer.decode(current_sentence))
        return sentences

    def chunk_text(self, text: str) -> List[str]:
        sentences = self.split_into_sentences(text)
        if len(sentences) <= self.window_size:
            return [text]

        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        window = sentences[:self.window_size]
        prev_embedding = self.get_embedding(" ".join(window))

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.tokenizer.encode(sentence)
            sentence_token_count = len(sentence_tokens)

            # Check if adding this sentence would exceed the token limit
            if current_chunk_tokens + sentence_token_count > self.max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_tokens = 0

            current_chunk.append(sentence)
            current_chunk_tokens += sentence_token_count

            # Only perform similarity check if we have enough sentences for a window
            if i >= self.window_size - 1:
                window = sentences[i-self.window_size+1:i+1]
                current_embedding = self.get_embedding(" ".join(window))

                similarity = self.cosine_similarity(prev_embedding, current_embedding)

                if similarity < self.threshold and len(current_chunk) > self.window_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_chunk_tokens = 0

                prev_embedding = current_embedding

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
