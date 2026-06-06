from src.dataset.subgraphs import (build_nx_graph, extract_bfs_subgraphs,
                                   extract_community_subgraphs, extract_module_subgraphs,
                                   save_subgraphs)
import ontolearner.ontology as ontology_module
import inspect
import os
from tqdm import tqdm

# configurations
depth = 2
min_nodes = 3
max_nodes = 20
resolution = 1.0
output_dir = "data"
strategies = [
    "bfs",
    # "community",
    # "module"
]

for name, obj in tqdm(inspect.getmembers(ontology_module), desc="Processing ontologies: "):
    if inspect.isclass(obj) and name != "BaseOntology":
        if hasattr(obj, 'load') and callable(getattr(obj, 'load')) and hasattr(obj, 'ontology_id'):
            if obj.ontology_id in ["ChEBI", "GO", "EFO", "NCIt", "OntoCAPE", "PRotein", "YAGO"]:
                continue

            ontology_name = obj.ontology_id
            ontology_dir = os.path.join(output_dir, ontology_name)
            if os.path.exists(ontology_dir):
                print(f"Skipping {ontology_name}, already processed.")
                continue

            ontology = obj()
            ontology.from_huggingface()

            print("Building undirected networkx graph...")
            my_nx_graph = build_nx_graph(ontology.rdf_graph)

            for strat in strategies:
                if strat == "bfs":
                    sgs = extract_bfs_subgraphs(ontology.rdf_graph,
                                                my_nx_graph,
                                                depth=depth,
                                                min_nodes=min_nodes,
                                                max_nodes=max_nodes)
                elif strat == "community":
                    sgs = extract_community_subgraphs(ontology.rdf_graph,
                                                      my_nx_graph,
                                                      min_nodes=min_nodes,
                                                      max_nodes=max_nodes,
                                                      resolution=resolution)
                elif strat == "module":
                    sgs = extract_module_subgraphs(ontology.rdf_graph,
                                                   my_nx_graph,
                                                   min_nodes=min_nodes,
                                                   max_nodes=max_nodes)
                if len(sgs) >= 1:
                    save_subgraphs(sgs, output_dir, ontology_name, strat)

            print(f"\nDone. All sub-graphs saved to: {output_dir}/{ontology_name}/")
