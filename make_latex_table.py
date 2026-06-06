import json
import os
import pandas as pd

RESULTS_DIR = "results"

METRICS = [
    "triplet_accuracy",
    "MRR",
    "Recall@1",
    "hard_negative_acc",
]

METRIC_NAMES = {
    "triplet_accuracy": "Triplet Acc",
    "MRR": "MRR",
    "Recall@1": "R@1",
    "hard_negative_acc": "Hard Neg Acc",
}

MODEL_NAMES = {
    "BAAI_bge-small-en-v1.5": "BGE-small",
    "BAAI_bge-base-en-v1.5": "BGE-base",
    "BAAI_bge-large-en-v1.5": "BGE-large",

    "intfloat_e5-small-v2": "E5-small",
    "intfloat_e5-base-v2": "E5-base",
    "intfloat_e5-large-v2": "E5-large",

    "nomic-ai_nomic-embed-text-v1.5": "Nomic-embed",
    "nomic-ai_nomic-embed-text-v2-moe": "Nomic-embed-MoE",

    "sentence-transformers_all-MiniLM-L6-v2": "MiniLM-L6",
    "sentence-transformers_all-mpnet-base-v2": "MPNET-base",
    "sentence-transformers_all-roberta-large-v1": "RoBERTa-large",

    "thenlper_gte-small": "GTE-small",
    "thenlper_gte-base": "GTE-base",
    "thenlper_gte-large": "GTE-large",

    "Qwen_Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "Qwen_Qwen3-Embedding-4B": "Qwen3-Embedding-4B",
    "Qwen_Qwen3-Embedding-8B": "Qwen3-Embedding-8B",
}

rows = []

for file in os.listdir(RESULTS_DIR):

    if not file.endswith(".json"):
        continue

    path = os.path.join(RESULTS_DIR, file)

    with open(path, "r") as f:
        results = json.load(f)

    model_key = file.replace(".json", "")
    model_name = MODEL_NAMES.get(model_key, model_key)

    row = {
        "Model": model_name
    }

    cross_domain = results["test"]
    for metric in METRICS:
        row[("Cross-Ontology", METRIC_NAMES[metric])] = round(
            cross_domain[metric], 3
        )

    rows.append(row)

df = pd.DataFrame(rows)

columns = ["Model"]

for split in ["Cross-Ontology"]:
    for metric in METRICS:
        columns.append((split, METRIC_NAMES[metric]))

df = df[columns]

desired_order = [
    "MiniLM-L6",
    "MPNET-base",
    "RoBERTa-large",

    "Nomic-embed",
    "Nomic-embed-MoE",

    "E5-small",
    "E5-base",
    "E5-large",

    "GTE-small",
    "GTE-base",
    "GTE-large",

    "BGE-small",
    "BGE-base",
    "BGE-large",

    "Qwen3-Embedding-0.6B",
    "Qwen3-Embedding-4B",
    "Qwen3-Embedding-8B",
]

df["sort_key"] = df["Model"].apply(
    lambda x: desired_order.index(x)
    if x in desired_order else 999
)

df = df.sort_values("sort_key").drop(columns="sort_key")

latex_table = df.to_latex(
    index=False,
    multicolumn=True,
    multicolumn_format="c",
    escape=False,
    float_format="%.3f",
    column_format="l" + "ccccc" + "ccccc"
)

with open("benchmark_results.tex", "w") as f:
    f.write(latex_table)

print(df)
print("\nLaTeX table saved to: benchmark_results.tex")