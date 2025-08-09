# Embedding Model Options for 1024 Dimensions

If you want to keep your existing 1024-dimension Pinecone index, here are models that produce 1024 dimensions:

## Option 1: Use OpenAI Embeddings (Recommended for 1024d)
- Model: `text-embedding-ada-002`
- Dimensions: 1536 (would need to truncate to 1024)
- Requires OpenAI API key

## Option 2: Use Cohere Embeddings
- Model: `embed-english-v3.0`
- Dimensions: 1024
- Requires Cohere API key

## Option 3: Custom Sentence Transformer
Some sentence transformer models that produce higher dimensions:
- `sentence-transformers/all-roberta-large-v1` (1024d)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768d)

## Recommendation
For best results with your current setup, I recommend **Option 1** - creating a new 768-dimension index, as:
- ✅ No additional API costs
- ✅ Better performance with sentence transformers
- ✅ Optimized for semantic search
- ✅ Works out of the box