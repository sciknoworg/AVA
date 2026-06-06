import json
import hashlib
import logging
from pathlib import Path
import networkx as nx
import community.community_louvain as community_louvain  # python-louvain
from rdflib import Graph, RDF, RDFS, OWL, URIRef, BNode
from rdflib.namespace import SKOS
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

STRUCTURAL_PROPS = {
    RDFS.subClassOf,
    OWL.equivalentClass,
    OWL.disjointWith,
    RDFS.domain,
    RDFS.range,
    RDFS.subPropertyOf,
    OWL.inverseOf,
    OWL.someValuesFrom,
    OWL.allValuesFrom,
    OWL.onProperty,
    OWL.unionOf,
    OWL.intersectionOf,
}

LABEL_PROPS = {RDFS.label, SKOS.prefLabel, SKOS.altLabel}
COMMENT_PROPS = {RDFS.comment, SKOS.definition, SKOS.scopeNote}

def get_comment(g: Graph, uri) -> str:
    """Return human-readable comment/definition for a URI."""
    for prop in COMMENT_PROPS:
        for comment in g.objects(uri, prop):
            return str(comment).strip()
    return ""

def get_clean_label(rdf_graph, uri: str, language='en') -> str:
    """
    Extracts the label for a given URI in the specified language from the RDF graph.
    If no valid label is found, returns None.
    """
    entity = URIRef(uri)
    labels = list(rdf_graph.objects(subject=entity, predicate=RDFS.label))
    for label in labels:
        if hasattr(label, 'language') and label.language == language:
            return str(label)
    if labels:
        first_label = str(labels[0])
        if len(first_label) > 3 and not first_label.startswith("http"):
            return first_label
    if "#" in uri:
        local_name = uri.split("#")[-1]
    elif "/" in uri:
        local_name = uri.split("/")[-1]
    else:
        local_name = uri
    return local_name

def get_label(g: Graph, uri) -> str:
    """Return human-readable label for a URI."""
    for prop in LABEL_PROPS:
        for label in g.objects(uri, prop):
            return str(label)
    # Fall back to local name
    return get_clean_label(g, uri)

def get_classes(g: Graph) -> list:
    """Return all owl:Class URIs (non-blank)."""
    classes = set()
    for s in g.subjects(RDF.type, OWL.Class):
        if isinstance(s, URIRef):
            classes.add(s)
    for s in g.subjects(RDF.type, RDFS.Class):
        if isinstance(s, URIRef):
            classes.add(s)
    return list(classes)

def get_properties(g: Graph) -> list:
    """Return all property URIs."""
    props = set()
    for t in [OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty]:
        for s in g.subjects(RDF.type, t):
            if isinstance(s, URIRef):
                props.add(s)
    return list(props)

def build_nx_graph(g: Graph) -> nx.Graph:
    """
    Build an undirected NetworkX graph from the ontology.
    Nodes = classes + properties.
    Edges = structural relations between them.
    """
    G = nx.Graph()

    # Add class nodes
    for cls in get_classes(g):
        G.add_node(str(cls), label=get_label(g, cls), comment=get_comment(g, cls), node_type="class")

    # Add property nodes
    for prop in get_properties(g):
        G.add_node(str(prop), label=get_label(g, prop), comment=get_comment(g, prop), node_type="property")

    # Add edges from structural triples
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            if p in STRUCTURAL_PROPS:
                if G.has_node(str(s)) and G.has_node(str(o)):
                    G.add_edge(str(s), str(o), relation=get_label(g, p))

        # Handle blank node restrictions: class → someValuesFrom → filler
        if isinstance(s, URIRef) and isinstance(o, BNode):
            filler = g.value(o, OWL.someValuesFrom) or g.value(o, OWL.allValuesFrom)
            prop   = g.value(o, OWL.onProperty)
            if filler and isinstance(filler, URIRef):
                if G.has_node(str(s)) and G.has_node(str(filler)):
                    rel_label = get_label(g, prop) if prop else "restriction"
                    G.add_edge(str(s), str(filler), relation=rel_label)

    log.info(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def _process_single_bfs_seed(seed_uri, nx_graph, depth, min_nodes, max_nodes, g_triples_fallback=None):
    """Helper function to run BFS on a single seed for multiprocessing."""
    seed = str(seed_uri)
    if seed not in nx_graph:
        return None

    # BFS up to `depth` hops
    nodes = set(nx.single_source_shortest_path_length(nx_graph, seed, cutoff=depth).keys())

    if not (min_nodes <= len(nodes) <= max_nodes):
        return None

    fp = _fingerprint(nodes)
    sub = nx_graph.subgraph(nodes).copy()

    return {
        "seed": seed,
        "seed_uri": seed_uri,
        "nodes": nodes,
        "fp": fp,
        "sub": sub
    }


def _process_single_community(comm_item, nx_graph, min_nodes, max_nodes):
    """Helper function to process a single community for multiprocessing."""
    comm_id, nodes = comm_item
    
    if not (min_nodes <= len(nodes) <= max_nodes):
        return None

    sub = nx_graph.subgraph(nodes).copy()
    
    return {
        "comm_id": comm_id,
        "nodes": nodes,
        "sub": sub
    }


def _process_single_module(comp_nodes, nx_graph, min_nodes, max_nodes):
    """Helper function to process a single component/module for multiprocessing."""
    comp = nx_graph.subgraph(comp_nodes).copy()

    if len(comp_nodes) <= max_nodes:
        # Component is small enough — use as-is
        chunks = [set(comp_nodes)]
    else:
        # Large component — subdivide around hierarchy roots
        chunks = _subdivide_by_roots(comp, max_nodes)
        
    results_for_comp = []
    
    for nodes in chunks:
        if len(nodes) < min_nodes:
            continue
        fp = _fingerprint(nodes)
        sub = nx_graph.subgraph(nodes).copy()
        
        # Find the most "central" node as the module root
        try:
            root = max(nodes, key=lambda n: nx.degree_centrality(sub).get(n, 0))
        except Exception:
            root = list(nodes)[0]

        results_for_comp.append({
            "fp": fp,
            "root": root,
            "nodes": nodes,
            "sub": sub
        })
        
    return results_for_comp


def extract_bfs_subgraphs(g: Graph, nx_graph: nx.Graph, depth: int = 2, min_nodes: int = 3, max_nodes: int = 30, cpu_counts: int = 20) -> list[dict]:
    log.info(f"BFS extraction: depth={depth}, min={min_nodes}, max={max_nodes}")
    subgraphs = []
    seen_fingerprints = set()
    classes = get_classes(g)

    num_workers = max(1, cpu_counts)
    pool = Pool(processes=num_workers)

    worker_fn = partial(_process_single_bfs_seed,
                        nx_graph=nx_graph,
                        depth=depth,
                        min_nodes=min_nodes,
                        max_nodes=max_nodes)

    log.info(f"Running BFS across {len(classes)} classes using {num_workers} parallel workers...")

    results = []
    for r in tqdm(pool.imap_unordered(worker_fn, classes), total=len(classes), desc="Parallel BFS Search"):
        if r is not None:
            results.append(r)

    pool.close()
    pool.join()

    for r in tqdm(results, desc="Extracting Triples from discovered BFS components"):
        fp = r["fp"]
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        nodes = r["nodes"]
        sub = r["sub"]
        seed = r["seed"]
        seed_uri = r["seed_uri"]

        triples = _extract_triples(g, nodes)

        subgraphs.append({
            "subgraph_id": f"bfs_{_short_hash(fp)}",
            "strategy":    "bfs",
            "seed":        seed,
            "seed_label":  get_label(g, seed_uri),
            "depth":       depth,
            "nodes":       list(nodes),
            "node_labels": {n: nx_graph.nodes[n].get("label", n) for n in nodes},
            "node_comments": {n: nx_graph.nodes[n].get("comment", "") for n in nodes if nx_graph.nodes[n].get("comment")},
            "edges":       list(sub.edges(data=True)),
            "triples":     triples,
            "triple_count": len(triples),
        })

    log.info(f"BFS: extracted {len(subgraphs)} sub-graphs")
    return subgraphs

def extract_community_subgraphs(g: Graph, nx_graph: nx.Graph, min_nodes: int = 3, max_nodes: int = 50, resolution: float = 1.0, cpu_counts: int = 20) -> list[dict]:
    log.info(f"Community detection: resolution={resolution}")

    if nx_graph.number_of_nodes() == 0:
        log.warning("Empty graph — skipping community detection")
        return []

    try:
        partition = community_louvain.best_partition(nx_graph, resolution=resolution)
    except TypeError as e:
        log.error(f"Louvain failed (graph might be directed or invalid): {e}")
        return []

    communities: dict[int, list] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    log.info(f"Found {len(communities)} communities")
    subgraphs = []

    num_workers = max(1, cpu_counts - 1)
    pool = Pool(processes=num_workers)
    
    worker_fn = partial(_process_single_community, 
                        nx_graph=nx_graph, 
                        min_nodes=min_nodes, 
                        max_nodes=max_nodes)

    log.info(f"Processing {len(communities)} communities using {num_workers} parallel workers...")
    
    results = []
    for r in tqdm(pool.imap_unordered(worker_fn, communities.items()), total=len(communities), desc="Parallel Community Prep"):
        if r is not None:
            results.append(r)
            
    pool.close()
    pool.join()

    for r in tqdm(results, desc="Extracting Triples from communities"):
        comm_id = r["comm_id"]
        nodes = r["nodes"]
        sub = r["sub"]

        triples = _extract_triples(g, set(nodes))

        subgraphs.append({
            "subgraph_id":  f"comm_{comm_id:04d}",
            "strategy":     "community",
            "community_id": comm_id,
            "nodes":        nodes,
            "node_labels":  {n: nx_graph.nodes[n].get("label", n) for n in nodes},
            "node_comments": {n: nx_graph.nodes[n].get("comment", "") for n in nodes if nx_graph.nodes[n].get("comment")},
            "edges":        list(sub.edges(data=True)),
            "triples":      triples,
            "triple_count": len(triples),
        })

    log.info(f"Community: extracted {len(subgraphs)} sub-graphs")
    return subgraphs

def extract_module_subgraphs(g: Graph, nx_graph: nx.Graph, min_nodes: int = 3, max_nodes: int = 40, cpu_counts: int = 20) -> list[dict]:
    log.info("Module extraction: connected components + hierarchy roots")
    subgraphs = []
    seen_fps  = set()

    components = list(nx.connected_components(nx_graph))
    if not components:
        log.warning("Empty graph — skipping module extraction")
        return []

    num_workers = max(1, cpu_counts - 1)
    pool = Pool(processes=num_workers)
    
    worker_fn = partial(_process_single_module, 
                        nx_graph=nx_graph, 
                        min_nodes=min_nodes, 
                        max_nodes=max_nodes)

    log.info(f"Processing {len(components)} components using {num_workers} parallel workers...")
    
    results = []
    for comp_results in tqdm(pool.imap_unordered(worker_fn, components), total=len(components), desc="Parallel Module Prep"):
        if comp_results:
            results.extend(comp_results)
            
    pool.close()
    pool.join()

    for r in tqdm(results, desc="Extracting Triples from modules"):
        fp = r["fp"]
        if fp in seen_fps:
            continue
        seen_fps.add(fp)

        nodes = r["nodes"]
        sub = r["sub"]
        root = r["root"]
        
        root_label = nx_graph.nodes[root].get("label", root)

        triples = _extract_triples(g, nodes)

        subgraphs.append({
            "subgraph_id":  f"mod_{_short_hash(fp)}",
            "strategy":     "module",
            "root":         root,
            "root_label":   root_label,
            "nodes":        list(nodes),
            "node_labels":  {n: nx_graph.nodes[n].get("label", n) for n in nodes},
            "node_comments": {n: nx_graph.nodes[n].get("comment", "") for n in nodes if nx_graph.nodes[n].get("comment")},
            "edges":        list(sub.edges(data=True)),
            "triples":      triples,
            "triple_count": len(triples),
        })

    log.info(f"Module: extracted {len(subgraphs)} sub-graphs")
    return subgraphs

def _subdivide_by_roots(comp: nx.Graph, max_nodes: int) -> list[set]:
    roots = sorted(comp.nodes, key=lambda n: comp.degree(n))
    chunks = []
    visited = set()

    for root in roots:
        if root in visited:
            continue

        nodes = set(nx.single_source_shortest_path_length(comp, root, cutoff=3).keys())
        nodes -= visited
        if len(nodes) >= 3:
            chunks.append(nodes)
            visited.update(nodes)

    return chunks

def _extract_triples(g: Graph, nodes: set) -> list[dict]:
    triples = []
    node_uris = {URIRef(n) for n in nodes}

    for s, p, o in g:
        if not isinstance(s, URIRef):
            continue

        if s in node_uris and isinstance(o, URIRef) and o in node_uris:
            triples.append({
                "subject":   str(s),
                "predicate": str(p),
                "object":    str(o),
                "subject_label":   _safe_label(g, s),
                "predicate_label": _safe_label(g, p),
                "object_label":    _safe_label(g, o),
                "type": "direct",
            })

        if s in node_uris and isinstance(o, BNode):
            filler = g.value(o, OWL.someValuesFrom) or g.value(o, OWL.allValuesFrom)
            prop   = g.value(o, OWL.onProperty)
            q_type = "someValuesFrom" if g.value(o, OWL.someValuesFrom) else "allValuesFrom"
            if filler and isinstance(filler, URIRef) and filler in node_uris:
                triples.append({
                    "subject":   str(s),
                    "predicate": str(prop) if prop else str(p),
                    "object":    str(filler),
                    "subject_label":   _safe_label(g, s),
                    "predicate_label": _safe_label(g, prop) if prop else _safe_label(g, p),
                    "object_label":    _safe_label(g, filler),
                    "type": f"restriction_{q_type}",
                })

    seen = set()
    unique = []
    for t in triples:
        key = (t["subject"], t["predicate"], t["object"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique

def _safe_label(g: Graph, uri) -> str:
    try:
        return get_label(g, uri)
    except Exception:
        return str(uri).split("#")[-1].split("/")[-1]

def _fingerprint(nodes: set) -> str:
    return ",".join(sorted(nodes))

def _short_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:8]

def save_subgraphs(subgraphs: list[dict], output_dir: str, ontology_name: str, strategy: str) -> None:
    """Save sub-graphs as individual JSON files + a manifest."""
    out = Path(output_dir) / ontology_name / strategy
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for sg in subgraphs:
        sg_id   = sg["subgraph_id"]
        sg_path = out / f"{sg_id}.json"

        # Add provenance
        sg["source_ontology"] = ontology_name
        sg["strategy"]        = strategy

        with open(sg_path, "w") as f:
            json.dump(sg, f, indent=2, default=str)

        manifest.append({
            "subgraph_id":   sg_id,
            "file":          str(sg_path),
            "triple_count":  sg["triple_count"],
            "node_count":    len(sg["nodes"]),
        })

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "ontology":    ontology_name,
            "strategy":    strategy,
            "total":       len(subgraphs),
            "subgraphs":   manifest,
        }, f, indent=2)

    log.info(f"Saved {len(subgraphs)} sub-graphs → {out}")
    log.info(f"Manifest → {manifest_path}")
