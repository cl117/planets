# Sequence-Aware Synthetic Part Graph GNN

This project trains a **sequence-aware graph neural network (GNN)** for synthetic biology part graphs. The model takes:

1. **Part DNA sequences** as graph nodes, such as promoters, RBSs, CDSs, terminators, operators, sensors, and output devices.
2. **Biological interactions** as directed graph edges, such as activation, repression, physical adjacency, or unknown regulatory links.

The goal is to learn representations that combine **DNA sequence semantics** with **genetic circuit topology**, then predict useful biological labels such as:

- **Part function**: promoter, RBS, CDS, terminator, regulator, sensor, output device, etc.
- **Interaction type**: activation, repression, physical adjacency, crosstalk, or unknown interaction.
- **Circuit behavior**: logic behavior, bistability, oscillation, dynamic response, or circuit-level success/failure.
- **Product/output**: fluorescence, RPU, expression level, or another scalar experimental readout.

The current implementation lives in [`synbio_seq_gnn.py`](synbio_seq_gnn.py). It uses a pretrained DNA sequence encoder, such as **DNABERT2** or **HyenaDNA**, followed by an edge-aware **GATv2** graph neural network.

---

## Project Workflow

```text
Public biological databases
        |
        v
Build sequence-aware graph data
        |
        v
Encode DNA parts with DNABERT2 / HyenaDNA
        |
        v
Run edge-aware GNN over part-interaction graph
        |
        v
Predict part function, interaction type, circuit behavior, or product output
        |
        v
Test on held-out public database records / benchmark circuits
```

---

## 1. Build Graph Data

The first step is to convert public synthetic biology and regulatory biology data into PyTorch Geometric graphs.

Each graph should contain:

```python
part_sequences = [
    "TTGACATATAAT",       # promoter sequence
    "AGGAGG",             # RBS sequence
    "ATGGCCATTGTAATGGG",  # CDS sequence
    "TTATTT",             # terminator-like sequence
]

interactions = [
    {"src": 0, "dst": 2, "type": "activate"},
    {"src": 2, "dst": 0, "type": "repress"},
    {"src": 1, "dst": 2, "type": "physical"},
]
```

The helper function `build_pyg_data(...)` converts this format into a `torch_geometric.data.Data` object.

### Recommended Public Data Sources

Use several complementary public databases because no single database perfectly contains DNA sequences, regulatory interactions, graph topology, and circuit-level behavior labels.

| Source | Use in this project | Possible labels |
| --- | --- | --- |
| **RegulonDB** | E. coli transcriptional regulation graph; useful for regulator-target edges and activation/repression labels. | edge labels, regulator function, promoter regulation |
| **SynBioHub / SBH** | Synthetic biology parts, designs, SBOL records, GenBank, and FASTA exports. | part sequence, part role, physical adjacency, design topology |
| **Cello / Cello UCF** | Genetic circuit design libraries and gate response information. | circuit logic, gate function, output response, product/RPU |
| **iGEM Registry / SynBioHub iGEM collections** | Community synthetic biology parts and constructs. | part role, part sequence, construct metadata |
| **BioModels / SBML models** | Curated dynamic models such as toggle switches and repressilators. | graph-level behavior labels, e.g. bistable or oscillatory |
| **DREAM GRN benchmarks** | Gene regulatory network inference benchmark data. | edge existence, link prediction, regulatory graph structure |

### Data Conversion Plan

A practical data-building pipeline is:

1. **Download source records**
   - RegulonDB: regulatory interactions, transcription factors, promoters, genes, evidence levels.
   - SynBioHub/SBH: SBOL, GenBank, or FASTA records for synthetic parts and designs.
   - Cello UCF: gate libraries, input sensors, output devices, and response functions.

2. **Normalize biological entities into graph nodes**
   - promoter
   - RBS
   - CDS/gene
   - terminator
   - transcription factor
   - sensor
   - reporter/output product

3. **Normalize interactions into directed graph edges**
   - `activate`: regulator increases target activity.
   - `repress`: regulator decreases target activity.
   - `physical`: part A is physically adjacent to part B in a construct.
   - `unknown`: interaction exists but type is missing or uncertain.

4. **Attach labels when available**
   - Node labels: part type or biological function.
   - Edge labels: activation, repression, crosstalk, or physical adjacency.
   - Graph labels: logic function, bistability, oscillation, or dynamic response.
   - Product labels: measured fluorescence, RPU, expression level, or yield.

---

## 2. Run the GNN

The model architecture is:

```text
DNA sequence per node
        |
        v
DNABERT2 / HyenaDNA sequence encoder
        |
        v
Node embedding projection
        |
        v
Typed-edge GATv2 message passing
        |
        v
Prediction heads
```

The current model supports four output heads:

| Output | Meaning | Example target |
| --- | --- | --- |
| `node_logits` | Node-level prediction | promoter/RBS/CDS/terminator classification |
| `edge_logits` | Edge-level prediction | activation/repression/physical/crosstalk prediction |
| `graph_logits` | Whole-graph prediction | bistable/oscillatory/dynamic circuit class |
| `product_pred` | Graph-level scalar prediction | GFP, RPU, yield, expression strength |

### Minimal Usage

```python
from synbio_seq_gnn import ModelConfig, SeqAwareGNN, build_pyg_data

cfg = ModelConfig(
    hf_model_name="zhihan1996/DNABERT-2-117M",
    node_out_dim=8,
    edge_out_dim=4,
    graph_out_dim=3,
    product_out_dim=1,
)

model = SeqAwareGNN(cfg)

data = build_pyg_data(
    part_sequences=part_sequences,
    interactions=interactions,
    graph_label=1,
    node_labels=[0, 1, 2, 3],
    edge_labels=[0, 1, 2],
    product_target=0.73,
)
```

To run the built-in demo:

```bash
python synbio_seq_gnn.py
```

To perform a fast syntax check only:

```bash
python -m py_compile synbio_seq_gnn.py
```

---

## 3. Test with Public Database Data

A recommended benchmark strategy is to test the project in three stages.

### Stage A: Part Function Prediction

**Goal:** predict part function from sequence and graph context.

- Source data: SynBioHub/SBH, iGEM parts, Cello parts.
- Input: part DNA sequence plus physical adjacency graph.
- Target: promoter, RBS, CDS, terminator, regulator, sensor, output device.
- Model output: `node_logits`.

### Stage B: Interaction Prediction

**Goal:** predict whether two biological entities interact and what type of interaction they have.

- Source data: RegulonDB, DREAM GRN benchmarks, Cello gates.
- Input: regulator/gene/promoter graph with DNA sequence features where available.
- Target: activation, repression, physical adjacency, crosstalk, or no interaction.
- Model output: `edge_logits`.

### Stage C: Circuit Behavior or Product Prediction

**Goal:** predict whole-circuit behavior or experimental output.

- Source data: Cello UCF, BioModels, curated toggle switch and repressilator examples.
- Input: full genetic circuit graph.
- Target: logic function, bistability, oscillation, dynamic response, fluorescence, RPU, or yield.
- Model output: `graph_logits` or `product_pred`.

---

## Suggested Repository Roadmap

The current repository contains the model scaffold. The next engineering steps are:

1. **Add dataset download scripts**
   - `data/download_regulondb.py`
   - `data/download_synbiohub.py`
   - `data/download_cello_ucf.py`

2. **Add parsers**
   - RegulonDB interaction table parser.
   - SBOL/GenBank/FASTA parser for SynBioHub/SBH records.
   - Cello UCF JSON parser.

3. **Add graph builders**
   - `build_regulondb_graph(...)`
   - `build_synbiohub_part_graph(...)`
   - `build_cello_circuit_graph(...)`

4. **Add training and evaluation scripts**
   - `train_node_function.py`
   - `train_interaction_prediction.py`
   - `train_circuit_behavior.py`
   - `evaluate_public_benchmark.py`

5. **Add benchmark splits**
   - train/validation/test split by part family.
   - train/validation/test split by circuit design.
   - held-out public database collection for external validation.

---

## Installation Notes

The core model expects:

```bash
pip install torch transformers torch-geometric
```

Depending on your machine and CUDA version, PyTorch Geometric may require installation instructions specific to your environment. See the official PyTorch Geometric installation guide before running large experiments.

---

## Current Status

Implemented:

- DNA sequence encoder wrapper using Hugging Face `AutoTokenizer` and `AutoModel`.
- Edge-aware GATv2 message passing.
- Relation types: `activate`, `repress`, `physical`, and `unknown`.
- Multi-task outputs for node, edge, graph, and product prediction.
- Minimal demo data builder and training step.

Not yet implemented:

- Public database download scripts.
- RegulonDB/SynBioHub/Cello parsers.
- Full benchmark training loop.
- Metrics and result tables.
- Reproducible dataset split files.

---

## References and Data Portals

- RegulonDB: <https://regulondb.ccg.unam.mx/>
- SynBioHub/SBH: <https://synbiohub.org/>
- SynBioHub API documentation: <https://wiki.synbiohub.org/api-docs/>
- Cello: <https://github.com/CIDARLAB/cello>
- Cello UCF Zenodo record: <https://zenodo.org/records/4675719>
- BioModels: <https://www.ebi.ac.uk/biomodels/>
- DREAM Challenges: <https://dreamchallenges.org/>
