# QueryAgent Reproduction — GrailQA

Reproduction of [QueryAgent (Huang et al., ACL 2024)](https://arxiv.org/abs/2403.11886) on the GrailQA dataset, with an updated relation ranking module using OpenAI `text-embedding-3-small` (1536d) embeddings in place of the original unreleased embedding files.

> **Original paper:** QueryAgent: A Reliable and Efficient Reasoning Framework with Environmental Feedback-based Self-Correction (ACL 2024)
> **Original repository:** https://github.com/cdhx/QueryAgent

---

## Overview

QueryAgent solves Knowledge Base Question Answering (KBQA) by decomposing a question into a sequence of atomic actions (get_relation, add_fact, add_count, etc.) that progressively construct a structured query (PyQL), which is then compiled to SPARQL and executed against a Freebase endpoint. A relation ranking module re-ranks candidate relations at each step using embedding-based cosine similarity.

This reproduction covers the **GrailQA** dataset only. 

---

## Requirements

- Python 3.10
- A Freebase SPARQL endpoint (Virtuoso recommended — see [Setup](#setup))
- An OpenAI API key (for LLM calls and optionally for embedding generation)

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The current `requirements.txt` has commented out NVIDIA/CUDA and vLLM dependencies for a standard environment. If you need GPU acceleration, simply uncomment those lines before running `pip install`. Core dependencies are: `openai`, `sentence-transformers`, `scikit-learn`, `SPARQLWrapper`, `neo4j`, `tqdm`, `torch`.

---

## Setup

### 1. Freebase SPARQL Endpoint

This reproduction requires a running Freebase SPARQL endpoint. Follow the setup instructions from the original GrailQA repository to deploy Freebase on Virtuoso:

https://github.com/dki-lab/GrailQA?tab=readme-ov-file#setup

Once Virtuoso is running, note your endpoint URL (e.g. `http://localhost:3001/sparql`).

If your endpoint is running on a remote server, set up an SSH tunnel to forward the port locally:

```bash
ssh -L <local_port>:localhost:<remote_port> <user>@<server> -N
```

Example:
```bash
ssh -L 3001:localhost:3001 user@your-server -N
```

Keep this terminal open while running experiments.

### 2. Embedding Files

The original paper's embedding files are not publicly released. Re-generate them using `text-embedding-3-small` (1536d):

```bash
# Generate relation embeddings
python3 ag_src/agent_utils/generate_relation_embed.py

# Generate question embeddings for GrailQA
python3 ag_src/agent_utils/generate_question_embed.py
```

Place the output files under `data/`:
```
data/
├── fb_relation_embed_1536.json
└── grailqa_question_embed_1536.json
```

Format: `{"relation_name": [0.123, 0.456, ...], ...}`

### 3. Configuration

Edit `ag_src/agent_utils/config.py`:

```python
# Required
all_key = ['your-openai-or-openrouter-api-key']
SPARQLPATH = 'http://localhost:3001/sparql'   # your Freebase endpoint

# Model settings
config = {
    'dataset': 'grailqa',
    'model': 'gpt-3.5-turbo',
    'api_base': 'https://api.openai.com/v1', 
    'openai_embedding': True,
    'sentence_transformer': False,
    'use_neo4j': False,
    'self_correction': True,
    'golden_el': False,
    TEST_LIMIT: 500                        # number of questions to evaluate
}
```

---

## Running Experiments

```bash
cd ag_src
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 agent_utils/run_exp.py
```

To run in the background (recommended for remote servers):

```bash
nohup python3 agent_utils/run_exp.py > ../logs/run.log 2>&1 &
```

Monitor progress:

```bash
tail -f ../logs/run.log
```

---

## Results

Logs and results are saved to the `logs/` directory as `.json` files, named by model and timestamp:

```
logs/grailqa_gpt-3.5-turboi_sc_el_openai_emb_MM_DD_HH_MM_SS.json
```

Running F1 and EM scores are printed to stdout after each question.

### Reproduction Results (GrailQA dev set)

| Backbone LLM | Ranking | Questions | F1 | EM |
|---|---|---|---|---|
| GPT-3.5-turbo (paper) | OpenAI Embedding (OE) | Full dev | 56.3 | — |
| GPT-3.5-turbo | OpenAI Embedding (OE) | 500 | ~57.0 | ~51.0 |


---

## Relation Ranking Modes

Controlled via `config.py`:

| Config | Mode | Description |
|---|---|---|
| `openai_embedding: True` | OE | Pre-computed OpenAI embedding cosine similarity (default) |
| `use_neo4j: False` | UDP | Neo4j vector index semantic ranking |


---

## Citation

```bibtex
@misc{huang2024queryagentreliableefficientreasoning,
      title={QueryAgent: A Reliable and Efficient Reasoning Framework with Environmental Feedback-based Self-Correction}, 
      author={Xiang Huang and Sitao Cheng and Shanshan Huang and Jiayu Shen and Yong Xu and Chaoyun Zhang and Yuzhong Qu},
      year={2024},
      eprint={2403.11886},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.11886}, 
}
```