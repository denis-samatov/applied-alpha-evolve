# Applied AlphaEvolve 🧬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Next-Gen LLMs track project, AIRI Summer School 2025*

A comparative study of three open-source LLM-driven evolutionary search frameworks —
OpenAlpha_Evolve, OpenEvolve, and MetaEvolve — benchmarked against Google's AlphaEvolve
on a circle-packing optimization task.

[**Report**](report.pdf) | [**Slides**](slides.pdf)

## About the project

Recent advances in automated solution search — exemplified by Google's
[AlphaEvolve framework](https://arxiv.org/pdf/2506.13131) — highlight the effectiveness
of combining large language models with evolutionary search to solve hard optimization
problems. AlphaEvolve sets a new standard in this space by pairing model-based candidate
generation with rigorous evaluation. In this study, we run a comparative analysis of
three open-source evolutionary frameworks — OpenAlpha_Evolve, OpenEvolve, and
MetaEvolve — on a circle-packing optimization task, using AlphaEvolve as the reference
benchmark. The analysis covers both framework architecture and the effect of different
LLMs. We also add logging and visualization tooling to better understand the
evolutionary process and how different models behave. The included report states that
MetaEvolve, using Qwen models, achieved the best solution quality among the open
frameworks studied (sum of radii = 2.6238), below the AlphaEvolve reference (2.635).
This work lays a foundation for further research into LLM-driven evolutionary
algorithms for optimization.

## 📁 Frameworks studied

#### [`MetaEvolve`](meta_evolve/) (closed-source repository)

> Description and benchmark results included; code has been modified and is still
> being improved.

- Multi-Island MAP-Elites algorithm implementation
- DAG-based program execution system
- Asynchronous architecture with Redis persistence
- Quality-diversity optimization support

#### [`OpenEvolve`](open_evolve/) ([upstream repository](https://github.com/codelion/openevolve))

> Description, benchmark results, and code with our modifications included.

- Implementation of AlphaEvolve concepts from Google DeepMind
- Automated code generation and optimization
- Web interface for monitoring the evolutionary process
- Multi-provider LLM support

#### [`OpenAlpha_Evolve`](open_alpha_evolve/) ([upstream repository](https://github.com/shyamsaktawat/OpenAlpha_Evolve))

> Description and benchmark results included; not selected for further development due
> to its limitations.

- Analysis of an open implementation of AlphaEvolve concepts
- Study of its agentic architecture and modular design
- Performance and scalability evaluation
- Comparative study of different approaches to evolutionary programming

## 🎯 Objectives

1. Study the architecture and internal logic of existing AlphaEvolve implementations
   (frameworks).
2. Test the system with different LLMs — from large proprietary models to smaller
   open ones — and evaluate its robustness and adaptability. Answer the question of
   whether this approach is viable for open-source solutions.
3. Identify possible improvements:
   - **Conceptual**: better mutation strategies, task-adapted metrics.
   - **Visual**: clearer visualization of the evolutionary process.
   - **Technical**: richer logging, solution-trajectory tracking.

## 🛠️ Tech stack

### Core technologies
- **Python 3.9+** — primary language (minimum declared by the vendored OpenEvolve package)
- **AsyncIO** — asynchronous programming
- **Redis** — high-performance state storage
- **Docker** — containerization and isolation

### Frameworks and libraries
- **Pydantic** — data validation and typing
- **Loguru** — structured logging
- **NumPy** — numerical computation
- **FastAPI** — web interfaces and APIs

### LLM integrations
- **Proprietary LLMs via API** — OpenAI, Mistral, Gemini
- **Local LLMs via vLLM** — Qwen, Llama, DeepSeek

## 🚀 Running it (OpenEvolve example)

System requirements
- Python 3.9+
- Docker (optional)
- vLLM (for running models locally)

LLM API access for the configuration below
- An OpenRouter-compatible API key. The configured models and provider may incur charges.

```bash
# Clone the repository
git clone https://github.com/denis-samatov/applied-alpha-evolve.git
cd applied-alpha-evolve

# Set up OpenEvolve
cd open_evolve
pip install -e .

# Configure a credential for the OpenRouter endpoint in config_phase_1.yaml.
# Do not commit the key or paste it into a YAML file.
export OPENAI_API_KEY='<provider-api-key>'

# Run phase 1 from the open_evolve directory.
python openevolve-run.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 100
```

This command is the actual CLI path in the repository. It requires a live LLM provider; the study results in the report were not regenerated as part of the repository audit. See the [circle-packing notes](open_evolve/examples/circle_packing/README.md) for the two-phase experiment and its checkpoint dependency.

## Licensing

This repository mixes licenses:

- The root [`LICENSE`](LICENSE) (MIT) covers this project's own analysis, `report.pdf`,
  `slides.pdf`, and the `meta_evolve/` and `open_alpha_evolve/` directories (descriptions
  and benchmark results — neither vendors code from its upstream).
- [`open_evolve/`](open_evolve/) is vendored from [codelion/openevolve](https://github.com/codelion/openevolve)
  with our modifications, and is licensed under **Apache-2.0** — see
  [`open_evolve/LICENSE`](open_evolve/LICENSE). The root MIT license does not apply to this
  directory.

See [`NOTICE`](NOTICE) for the full breakdown.
