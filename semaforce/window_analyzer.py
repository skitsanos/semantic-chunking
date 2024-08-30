import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
from typing import List, Tuple, Optional
from statistics import mean, stdev
import logging


class SemanticWindowAnalyzer:
    def __init__(
            self,
            model_name: str = 'all-MiniLM-L6-v2',
            model: str = "gpt-4o",
            cache_folder: str = '.cache'
    ):
        self.sentence_model = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.tokenizer = tiktoken.encoding_for_model(model)
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def split_into_sentences(
            self,
            text: str
    ) -> List[str]:
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

    def get_embedding(
            self,
            text: str
    ) -> np.ndarray:
        return self.sentence_model.encode([text], show_progress_bar=False)[0]

    def cosine_similarity(
            self,
            a: np.ndarray,
            b: np.ndarray
    ) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def calculate_similarity_distribution(
            self,
            sentences: List[str]
    ) -> List[float]:
        embeddings = [self.get_embedding(sentence) for sentence in sentences]
        similarities = [
            self.cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        return similarities

    def analyze_window_sizes(
            self,
            text: str,
            min_window: int = 2,
            max_window: int = 10,
            score_threshold: float = 1e-6
    ) -> Tuple[Optional[int], float, List[Tuple[int, float]]]:
        sentences = self.split_into_sentences(text)
        self.logger.debug(f"Number of sentences: {len(sentences)}")

        if len(sentences) <= max_window:
            return len(sentences), 0.0, [(len(sentences), 0.0)]

        similarities = self.calculate_similarity_distribution(sentences)
        self.logger.debug(f"Similarity distribution: {similarities}")

        window_scores = []

        for window_size in range(min_window, min(max_window, len(sentences)) + 1):
            windowed_similarities = [
                mean(similarities[i:i + window_size - 1])
                for i in range(0, len(similarities) - window_size + 2)
            ]
            self.logger.debug(
                f"Window size {window_size}, windowed similarities: {windowed_similarities}")

            if len(windowed_similarities) > 1:
                score = stdev(windowed_similarities)
                window_scores.append((window_size, score))
                self.logger.debug(f"Window size {window_size}, score: {score:.10f}")
            else:
                window_scores.append((window_size, float('inf')))
                self.logger.debug(f"Window size {window_size}, score: inf (not enough data)")

        if not window_scores:
            self.logger.warning("No valid window sizes found")
            return None, float('inf'), []

        # Sort by score (lower is better)
        window_scores.sort(key=lambda
            x: x[1])

        # Find the smallest window size within the threshold of the best score
        best_score = window_scores[0][1]
        best_window = None
        for window_size, score in window_scores:
            if score <= best_score + score_threshold:
                best_window = window_size
                break

        if best_window is None:
            self.logger.warning("No window size found within the specified threshold")
            best_window = window_scores[0][0]  # Default to the window size with the best score

        self.logger.info(f"Best window size: {best_window}, score: {best_score:.10f}")
        self.logger.info(f"All window scores: {[(w, f'{s:.10f}') for w, s in window_scores]}")

        return best_window, best_score, window_scores
