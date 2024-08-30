# Semantic Text Processing Tools

This repository contains two main classes for semantic text processing: `SemanticChunker` and `SemanticWindowAnalyzer`. These tools are designed to help with intelligent text splitting and analysis, particularly useful for large language model applications.

## SemanticChunker

`SemanticChunker` is a sophisticated tool for splitting large texts into semantically coherent chunks while respecting token limits.

### Features:
- Splits text based on semantic similarity and token count
- Uses SentenceTransformer for generating sentence embeddings
- Employs a sliding window approach for comparing semantic similarity
- Ensures chunks do not exceed a specified token limit
- Customizable parameters for fine-tuning

### Basic Usage:
```python
from chunker import SemanticChunker

chunker = SemanticChunker(
    model_name='all-MiniLM-L6-v2',
    window_size=3,
    threshold=0.2,
    max_tokens=8192,
    model="gpt-4"
)

text = "Your long text goes here..."
chunks = chunker.chunk_text(text)
```

## SemanticWindowAnalyzer

`SemanticWindowAnalyzer` is a tool for analyzing the optimal window size for processing text based on semantic similarity between sentences.

### Features:
- Splits text into sentences
- Calculates semantic similarity between adjacent sentences
- Analyzes various window sizes to find the optimal one
- Provides detailed logging for analysis steps
- Customizable parameters for different use cases

### Basic Usage:
```python
from window_analyzer import SemanticWindowAnalyzer

analyzer = SemanticWindowAnalyzer(
    model_name='all-MiniLM-L6-v2',
    model="gpt-4",
    cache_folder='.cache'
)

text = "Your text for analysis goes here..."
best_window, best_score, all_scores = analyzer.analyze_window_sizes(text)
```

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install numpy sentence-transformers tiktoken
   ```

## Note

Both tools use the `sentence-transformers` library and require a working Python environment with access to the internet for downloading pre-trained models.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](link-to-your-issues-page) if you want to contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)