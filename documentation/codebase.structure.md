
# `llm_perf` Codebase Architecture Overview

`llm_perf` is a small, modular toolkit for **Transformer/LLM inference performance modeling**.  
It’s organized around:

- **Specs**: typed configs for models, systems, partitions, and tuners  
- **Core models**: memory, FLOPs, traffic, communication, latency  
- **IO**: JSON loaders + a simple on-disk database  
- **Calculators**: a façade to run the full pipeline  
- **Utils**: equations and adapters (e.g., HuggingFace → `llm_perf.model.v1`)  

Below is the high-level layout and the role of each module.

---

## 1. Repository Layout (Nov 2025)

```text
.
├── quickstart.ipynb                  # end-to-end tutorial notebook
├── documentation/
│   ├── codebase.structure.md
│   ├── equations.cheetsheet.{md,pdf}
│   └── modeling.methodology.{md,pdf}
└── llm_perf/
    ├── __init__.py
    ├── calculators/
    │   └── inference_calculator.py
    ├── core/
    │   ├── comm_model.py
    │   ├── flops_model.py
    │   ├── latency_model.py
    │   ├── memory_model.py
    │   └── traffic_model.py
    ├── database/
    │   ├── model/
    │   │   ├── example.model.dense.json
    │   │   ├── example.model.moe.json
    │   │   ├── qwen3_vl_235b_fp8.json
    │   │   └── external.model/hf/
    │   │       └── qwen3_vl_235b_a22b_thinking_fp8.json
    │   ├── partition/example.partition.json
    │   ├── system/example.sys.json
    │   └── tuner/example.tuner.json
    ├── io/
    │   ├── database_loaders.py
    │   ├── model_loaders.py
    │   ├── partition_loaders.py
    │   ├── system_loaders.py
    │   └── tuner_loaders.py
    ├── specs/
    │   ├── model_spec.py
    │   ├── partition_spec.py
    │   ├── system_spec.py
    │   └── tuner_spec.py
    └── utils/
        ├── constants.py
        ├── data_check.py
        ├── equations.py
        └── hf_model_adapter.py
```

---

## 2. `llm_perf/__init__.py`

Public entrypoint for the package. It re-exports:

- `LlmModelSpec`, `MoESpec`
- `DeviceSpec`, `NetworkDomainSpec`, `SystemSpec`
- `PartitionSpec`
- `TuningSpec`
- `InferenceCalculator`, `InferenceResults`

Typical usage:

```python
from llm_perf import (
    LlmModelSpec, MoESpec,
    DeviceSpec, NetworkDomainSpec, SystemSpec,
    PartitionSpec, TuningSpec,
    InferenceCalculator,
)
```

---

## 3. Specs: `llm_perf/specs/`

Specs are **dataclasses** defining the inputs for the performance equations.

### 3.1 `model_spec.py`

- **`MoESpec`**  
  Fields:
  - `n_experts`, `k_active`, `I_moe`, `n_moe_layers`

- **`LlmModelSpec`**  
  Contains:
  - `L`, `H`, `n_q`, `n_kv`, `I_dense`, `vocab_size`
  - `max_seq_len`
  - `bytes_per_param`
  - Optional `moe: MoESpec`
  - Helpers: `d_head()`, `H_kv()`

### 3.2 `system_spec.py`

- **`DeviceSpec`**
  - `name`
  - `hbm_capacity_GB` (decimal GB)
  - `hbm_bandwidth_GBps` (decimal GB/s)
  - `peak_flops_TF`

- **`NetworkDomainSpec`**
  - `alpha_us` (latency)
  - `bandwidth_GBps` (GB/s decimal)

- **`SystemSpec`**
  - `device: DeviceSpec`
  - `num_devices`
  - `network_domains: Dict[str, NetworkDomainSpec]`

### 3.3 `partition_spec.py`

Defines the *structural* parallel layout:

- `DP`, `PP`, `TP`, `EP`, `SP`

### 3.4 `tuner_spec.py`

Defines all **execution/tuning knobs**:

- `S_decode`
- Algorithms: `tp_algorithm`, `ep_algorithm`
- Number of collectives/layer/token: `n_TP_collectives`, `n_EP_collectives`, `n_SP_collectives`
- Heuristics:
  - `c_act`
  - `flash_attn_gain` ($γ_{FA}$ ≥ 1)
  - `overlap_factor` (ρ - local compute and network latency overlap ratio)

---

## 4. Core Models: `llm_perf/core/`

### 4.1 `memory_model.py`

Produces `MemoryResults`:
- `M_theta_device`, `M_act_device`, `M_kv_device`
- `M_total_device`
- `fits_in_HBM`

Inputs: model, system, partition, tuner.

### 4.2 `flops_model.py`

Produces `FlopsResults`:
- `F_token_device`
- `F_layer_per_device`

### 4.3 `traffic_model.py`

Produces `TrafficResults`:
- `T_theta`, `T_act`, `T_kv`, `T_token_eff`

Incorporates:
- MoE
- flash attention gain
- activation scaling

### 4.4 `comm_model.py`

Produces `CommResults`:
- `t_PP`, `t_TP`, `t_EP`, `t_SP`, `t_comm_stage`

Implements:
- ring/tree TP all-reduce
- ring/tree EP all-to-all
- SP collectives
- pipeline stage aggregation

### 4.5 `latency_model.py`

Produces `LatencyResults`:
- `t_compute`, `t_mem`, `t_local`
- `t_comm`, `t_token`
- `TPS_single`, `TTPS`

Implements:
- compute/memory-bound time
- overlap factor (ρ)
- DP scaling

---

## 5. Calculator: `llm_perf/calculators/`

### `inference_calculator.py`

High-level flow:

```python
calc = InferenceCalculator(model, system, partition, tuner)
res  = calc.run()
```

Runs:
1. memory model  
2. flops model  
3. traffic model  
4. comm model  
5. latency model  

Packages results into `InferenceResults`.

---

## 6. IO: `llm_perf/io/`

### 6.1 `model_loaders.py`  
Loads `"llm_perf.model.v1"` → `LlmModelSpec`

### 6.2 `system_loaders.py`  
Loads `"llm_perf.system.v1"` → `SystemSpec`

### 6.3 `partition_loaders.py`  
Loads `"llm_perf.partition.v1"` → `PartitionSpec`

### 6.4 `tuner_loaders.py`  
Loads `"llm_perf.tuner.v1"` → `TuningSpec`

### 6.5 `database_loaders.py`  
Convenient database access:
- `/model/`
- `/system/`
- `/partition/`
- `/tuner/`

Functions:
- `list_model_ids()`, `load_model_from_db()`
- `list_hw_system_ids()`, `load_system_from_db()`
- `list_partition_ids()`, `load_partition_from_db()`
- `list_tuner_ids()`, `load_tuner_from_db()`

---

## 7. Utils: `llm_perf/utils/`

### 7.1 `equations.py`
Holds LaTeX-formatted strings of all equations.

### 7.2 `hf_model_adapter.py`
Converts HF `config.json` → `llm_perf.model.v1`.

Supports:
- text_config nesting
- Qwen/Qwen-VL
- MoE extraction
- fp8/bf16 dtype inference

### 7.3 `plotting.py`
Placeholder for future visualization tools.

---

## 8. Database: `llm_perf/database/`

Directory layout:

```
llm_perf/database/
  model/
    *.json
    external.model/hf/*.json
  system/
    *.json
  partition/
    *.json
  tuner/
    *.json
```

Provides canonical model/system configs for rapid experimentation.

---

## 9. Tutorials & Examples

- `quickstart.ipynb` — step-by-step walkthrough of ID discovery, optional HF conversion, and `InferenceCalculator` diagnostics
- Inline doc snippets (this file, `documentation/modeling.methodology.md`, etc.) for conceptual coverage

Standalone `test_*.py` drivers were removed to keep the repo lean; rely on the notebook or your own scripts for validation.

---

## 10. End-to-End Usage

```python
from llm_perf import InferenceCalculator
from llm_perf.io import (
    load_model_from_db,
    load_system_from_db,
    load_partition_from_db,
    load_tuner_from_db,
)

model = load_model_from_db("qwen3_vl_235b")
system = load_system_from_db("example")
partition = load_partition_from_db("example")
tuner = load_tuner_from_db("example")

calc = InferenceCalculator(model, system, partition, tuner)
res = calc.run()
```

Inspect:
- memory footprint (GB)
- compute/memory bound time
- comm time
- TTPS / TPS breakdown

---

This document summarizes the **hierarchy, responsibilities, and relationships** among modules in the `llm_perf` codebase.
