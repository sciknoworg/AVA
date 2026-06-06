# Datasets

This project uses two standard knowledge graph datasets for triple classification experiments:

- PWC21
- UMLS

---

## PWC21

**Description**

PWC21 is derived from *Papers With Code* and converted into RDF format.  
It focuses on evaluation results of research papers and their relationships.

- Entities: 192,115  
- Relations: 26  
- Triples: 284,875  

**Reference Paper**

Evaluation results can be found in:  
https://arxiv.org/pdf/2111.11845 (Table 4)

**Dataset Source**

Structured dataset available at:  
https://github.com/YaserJaradeh/exBERT/tree/main/data/PWC21

---

## UMLS

**Description**

UMLS (Unified Medical Language System) is a biomedical ontology dataset.  
It represents structured medical knowledge including concepts and relationships.

- Entities: 135  
- Relations: 46  
- Triples: 6,529  

**Reference Paper**

Evaluation results can be found in:  
https://arxiv.org/pdf/2111.11845 (Table 4)

**Dataset Sources**

Primary source:  
https://github.com/YaserJaradeh/exBERT/tree/main/data/UMLS  

Alternative source:  
https://pykeen.readthedocs.io/en/stable/api/pykeen.datasets.UMLS.html  

---

## Notes

- This repository does **not include datasets**.
- Please download them from the links above.
