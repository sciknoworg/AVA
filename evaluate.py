# --------------------------------------------------------------------------------
import os
import ssl
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
# --------------------------------------------------------------------------------
import json
import os
from sentence_transformers.sentence_transformer import SentenceTransformer
from src.evaluation import run_evaluation_report


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


data_files = {
    "test": "data/test.json",
    # "full": "data/synthesis.json"
}

models_to_evaluate = [
    # openai models
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",

    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-roberta-large-v1",

    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v2-moe",

    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",

    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",

    "thenlper/gte-small",
    "thenlper/gte-large",
    "thenlper/gte-base",

    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",

    "google/embeddinggemma-300m",
    "nvidia/llama-embed-nemotron-8b",
    "Linq-AI-Research/Linq-Embed-Mistral",
    "intfloat/multilingual-e5-large-instruct",

    "assets/MiniLM-triplet-m0.3/final",
    "assets/MiniLM-triplet-m0.5/final",
    "assets/MiniLM-triplet-m0.7/final",

    "assets/MiniLM-triplet-hyperbolic-m0.3-c0.1/final",
    "assets/MiniLM-triplet-hyperbolic-m0.3-c0.3/final",
    "assets/MiniLM-triplet-hyperbolic-m0.3-c0.5/final",
    "assets/MiniLM-triplet-hyperbolic-m0.3-c1.0/final",

    "assets/MiniLM-triplet-dpo-m0.3-beta0.1/final",
    "assets/MiniLM-triplet-dpo-m0.3-beta0.3/final",
    "assets/MiniLM-triplet-dpo-m0.3-beta0.5/final",

    "assets/MiniLM-triplet-hyperbolic-dpo-m0.3-c0.3-beta0.5-w_dpo0.3-w_hyperbolic0.7/final"
]


datasets = {}
for split_name, filepath in data_files.items():
    data = load_data(filepath)
    datasets[split_name] = data

os.makedirs("results", exist_ok=True)

for model_name in models_to_evaluate:
    print(f"\n\n{'=' * 50}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'=' * 50}")

    is_openai_model = False

    try:

        if model_name.startswith("text-embedding-"):
            is_openai_model = True
            model = model_name
        elif "Qwen/Qwen3-Embedding" in model_name or "thenlper/gte" in model_name:
            model_kwargs = {
                "trust_remote_code": True,
                "model_kwargs": {"attn_implementation": "eager"}
            }
            model = SentenceTransformer(model_name, **model_kwargs)
        else:
            model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        continue

    query_prefix = ""
    doc_prefix = ""

    if "intfloat/e5" in model_name:
        query_prefix = "query: "
        doc_prefix = "passage: "

    # elif "BAAI/bge" in model_name:
    #     query_prefix = "Represent this sentence for searching relevant passages: "
    #     # doc_prefix is empty for BGE
    # elif "nomic-ai" in model_name:
    #     query_prefix = "search_document: "
    #     doc_prefix = "search_document: "

    elif "Qwen/Qwen3-Embedding" in model_name or "Alibaba-NLP/gte-Qwen2" in model_name:
        # A standard retrieval instruction formulation for newer LLM-based embeddings
        query_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

    elif "tencent/KaLM" in model_name or "nvidia/llama" in model_name or "intfloat/multilingual" in model_name or "Linq-AI-Research" in model_name:
        query_prefix = "Instruct: Represent this sentence for searching relevant passages: \nQuery: "

    model_results = {}
    best_threshold = 0.5

    for split_name, data in datasets.items():
        if not data or split_name == "validation":
            continue

        print(f"\n--- Evaluating on split: {split_name} ---")

        metrics = run_evaluation_report(model, data,
                                        batch_size=128,
                                        classification_threshold=best_threshold,
                                        is_openai_model=is_openai_model,
                                        doc_prefix=doc_prefix,
                                        query_prefix=query_prefix)
        model_results[split_name] = metrics

    safe_model_name = model_name.replace("/", "_")
    results_path = os.path.join("results", f"{safe_model_name}.json")

    with open(results_path, 'w') as f:
        json.dump(model_results, f, indent=4)

    print(f"\nSaved results for {model_name} to {results_path}")
