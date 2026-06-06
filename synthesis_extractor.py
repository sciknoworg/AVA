import os
import json
from openai import OpenAI
import uuid
import time
import glob
from dotenv import find_dotenv, load_dotenv
from src.dataset.synthesis import load_synthesizer_llm, synthesize_data

_ = load_dotenv(find_dotenv())

def get_in_progress_ontologies():
    in_progress = set()
    for f in glob.glob("worker_*.json"):
        try:
            with open(f, "r") as wf:
                data = json.load(wf)
                if data.get("working_on"):
                    in_progress.add(data["working_on"])
        except Exception:
            pass
    return in_progress

synthesizer_model_id = "Qwen/Qwen3.5-35B-A3B"
openai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
openai_model = "gpt-4o-mini"
synthesizer_model, synthesizer_tokenizer = load_synthesizer_llm(model_id=synthesizer_model_id)

worker_id = str(uuid.uuid4())
worker_file = f"worker_{worker_id}.json"

try:
    root_dir = "data"
    if os.path.exists(root_dir):
        ontologies = [o for o in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, o)) and 'bfs' in os.listdir(os.path.join(root_dir, o))]

        for ontology in ontologies:

            out_file = os.path.join(root_dir, ontology, "bfs_synthetic_triplets.json")
            if os.path.exists(out_file):
                print(f"Skipping {ontology}, already completed.")
                continue

            if ontology in get_in_progress_ontologies():
                print(f"Skipping {ontology}, currently claimed by another worker.")
                continue

            with open(worker_file, "w") as wf:
                json.dump({"working_on": ontology}, wf)

            time.sleep(10)

            print(f"Worker {worker_id} claimed ontology: {ontology}")

            synthesize_data(ontology=ontology,
                            model=synthesizer_model, tokenizer=synthesizer_tokenizer,
                            openai_client=openai_client, openai_model=openai_model)

            with open(worker_file, "w") as wf:
                json.dump({"working_on": None}, wf)
finally:
    if os.path.exists(worker_file):
        os.remove(worker_file)
