# 🛍️ Text to Product Matching via Embeddings

A neural semantic search system leveraging transformer-based sentence embeddings for fashion product retrieval. Implements a hybrid ranking mechanism combining dense vector similarity with lexical feature overlap for robust query-product alignment.

## ✨ Features

- **Dense Retrieval**: Employs BAAI/bge-small-en-v1.5 transformer model for contextual embedding generation
- **Hybrid Ranking**: Ensemble approach fusing cosine similarity scores with token-level Jaccard coefficients
- **Adaptive Weighting**: Query-length-dependent interpolation parameter for optimal semantic-lexical balance
- **Low Latency Inference**: Sub-25ms retrieval after model warm-up with L2-normalized embeddings
- **Confidence Calibration**: Threshold-based fallback mechanism for low-confidence predictions

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy sentence-transformers scikit-learn matplotlib
```

### Usage

```python
# Semantic retrieval with hybrid scoring
results, latency_ms, fallback = rank("cozy minimalist neutral", top_k=3)

# Inspect ranked results with similarity decomposition
print(results)
```

## 🏗️ System Architecture

### Corpus Representation

Products are indexed with multi-modal metadata:
- **Textual Features**: Natural language descriptions
- **Categorical Tags**: Style attributes for lexical matching
- **Structured Metadata**: Category, tone, material taxonomies

### Text Encoding Pipeline

```python
# Multi-field fusion for enriched semantic representations
fusion_text = f"{description} tags: {' '.join(tags)}"
embeddings = encoder.encode(fusion_text, normalize_embeddings=False)
embeddings = L2_normalize(embeddings)  # Unit hypersphere projection
```

**Embedding Specifications:**
- **Model**: BAAI/bge-small-en-v1.5 (384-dimensional dense vectors)
- **Pooling**: Mean pooling over token embeddings
- **Normalization**: L2 normalization for cosine similarity optimization

### Retrieval & Ranking Algorithm

**Hybrid Score Function:**

```
S_hybrid(q, d) = (1 - α) × cos_sim(φ(q), φ(d)) + α × jaccard(tokens(q), tags(d))
```

Where:
- **φ(·)**: Sentence embedding function
- **cos_sim**: Cosine similarity in embedding space
- **α ∈ [0.10, 0.25]**: Dynamically computed interpolation weight
- **jaccard**: Token-level set overlap coefficient

**Adaptive Weighting Mechanism:**
- Short queries (≤3 tokens): Increase lexical weight (α = 0.20–0.25)
- Long queries (≥8 tokens): Prioritize semantic similarity (α = 0.10)
- Medium queries: Balanced weighting (α = 0.15)

## 📊 Performance Benchmarks

### Retrieval Quality

| Query | Top Match | Combined Score | Semantic Score | Lexical Score |
|-------|-----------|----------------|----------------|---------------|
| "energetic urban chic" | Techwear Jacket | 0.6036 | 0.7045 | 0.2000 |
| "cozy minimalist neutral" | Cashmere Sweater | 0.6511 | 0.7639 | 0.2000 |
| "vintage boho festival" | Boho Dress | 0.5834 | 0.6793 | 0.2000 |

### Latency Analysis

- **Cold Start**: ~1071ms (includes model initialization & first inference)
- **Warm Inference**: ~24ms average per query
- **Throughput**: ~40 QPS (queries per second) on CPU
- **Embedding Dimension**: 384-d (trade-off between expressiveness & speed)

### System Specifications

- **Corpus Size**: 7 products (extensible architecture)
- **Embedding Model**: 33M parameters
- **Memory Footprint**: ~130MB (model) + ~10KB (product embeddings)

## 🎯 NLP Techniques Employed

### 1. **Transfer Learning**
Pre-trained transformer encoders capture semantic nuances without task-specific fine-tuning.

### 2. **Dense Passage Retrieval (DPR)**
Bi-encoder architecture for efficient asymmetric query-document matching.

### 3. **Feature Fusion**
Concatenative approach combining unstructured text with structured tags for richer representations.

### 4. **Ensemble Ranking**
Late fusion of neural (embedding-based) and statistical (token-overlap) similarity signals.

### 5. **Vector Quantization**
L2 normalization enables efficient MIPS (Maximum Inner Product Search) via cosine similarity.

## 🔧 Configuration & Hyperparameters

### Model Selection

```python
PRIMARY_MODEL = "BAAI/bge-small-en-v1.5"       # Optimized for semantic search
FALLBACK_MODEL = "paraphrase-MiniLM-L12-v2"   # Lightweight alternative
```

**Model Comparison:**
- **BGE-small**: Better semantic understanding, 384-d embeddings
- **MiniLM**: Faster inference, 384-d embeddings, lower memory

### Tuning Parameters

- `top_k`: Retrieval depth (default: 3)
- `low_conf_threshold`: Confidence floor for fallback triggering (default: 0.35)
- `vibe_boost` (α): Lexical weight, query-adaptive ∈ [0.10, 0.25]

## 📈 Evaluation Framework

### Metrics

- **MRR (Mean Reciprocal Rank)**: Measures ranking quality
- **Precision@K**: Top-K retrieval accuracy
- **Score Distribution**: Confidence calibration analysis
- **Latency P50/P95**: Inference time percentiles

### Ablation Study Potential

- Semantic-only (α = 0) vs. Lexical-only (α = 1) vs. Hybrid
- Impact of text fusion strategy on retrieval quality
- Effect of embedding dimensionality on speed-accuracy trade-off

## 🛣️ Future Enhancements

### Model Improvements
- [ ] Fine-tune embeddings on fashion-specific corpus (domain adaptation)
- [ ] Implement cross-encoders for re-ranking top-K results
- [ ] Explore contrastive learning with triplet loss

### System Optimizations
- [ ] Approximate nearest neighbor search (FAISS/Annoy) for large-scale retrieval
- [ ] Query understanding module (intent classification, entity extraction)
- [ ] Multi-modal embeddings incorporating product images (CLIP-based)

### Evaluation
- [ ] Human relevance judgments for ground truth
- [ ] A/B testing framework for online evaluation
- [ ] Diversity metrics (avoiding redundant results)

## 📚 Technical Stack

**NLP Libraries:**
- `sentence-transformers`: Transformer-based embedding generation
- `scikit-learn`: Cosine similarity computation & normalization

**Core Dependencies:**
- `numpy`: Vector operations & numerical computation
- `pandas`: Structured data handling

## 📖 References

- Reimers & Gurevych (2019): Sentence-BERT for semantic textual similarity
- Karpukhin et al. (2020): Dense Passage Retrieval for open-domain QA
- Xiao et al. (2023): C-Pack: Comprehensive training for retrieval models

## 📝 License

MIT License - Open for research and commercial use

## 🤝 Contributing

Contributions welcome! Priority areas:
- **Corpus Expansion**: Diverse product categories & larger scale
- **Query Normalization**: Synonym expansion, spelling correction
- **Personalization**: User preference modeling with contextual embeddings
- **Multilingual Support**: Cross-lingual retrieval capabilities

---
