<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# ARAG: Agentic Retrieval-Augmented Generation for Personalized Recommendation

**Complexity:** Intermediate

This example demonstrates how to build a multi-agent recommendation pipeline that combines **parallel** and **sequential** execution in a single NeMo Agent Toolkit workflow. It introduces a new `parallel_executor` control flow component that enables fan-out/fan-in patterns alongside the existing `sequential_executor`.

## Table of Contents

- [Architecture](#architecture)
- [New Component: parallel\_executor](#new-component-parallel_executor)
- [Configuration](#configuration)
- [Installation and Setup](#installation-and-setup)
- [Run the Workflow](#run-the-workflow)

## Architecture

ARAG uses four specialized agents orchestrated in a hybrid parallel-sequential pipeline:

```
                         ┌──→ NLI Agent → Context Summary Agent ──┐
User Query → RAG Retriever ──┤                                          ├──→ Item Ranker → Ranked Items
                         └──→ User Understanding Agent ────────────┘
```

| Step | Agent | Description |
|------|-------|-------------|
| 1 | **RAG Retriever** | Retrieves an initial recall set of candidate items |
| 2a | **NLI Agent** | Evaluates semantic alignment (entailment) between candidates and user intent |
| 2b | **Context Summary Agent** | Summarizes NLI findings into concise context |
| 2c | **User Understanding Agent** | Infers user preferences from session context (runs **in parallel** with 2a-2b) |
| 3 | **Item Ranker Agent** | Produces a final ranked list by combining NLI context and user preference signals |

Steps 2a-2b form a sequential sub-pipeline. Step 2c runs concurrently with that sub-pipeline. All of this is expressed purely in the config file through composition.

## New Component: parallel_executor

The `parallel_executor` is a new control flow component that complements the existing `sequential_executor`. It enables **fan-out/fan-in** patterns where:

- Every function in the `tool_list` receives the **same input** (broadcast from the previous step)
- All functions execute **concurrently** via `asyncio.gather`
- Outputs are merged into a JSON object keyed by function name

### Configuration

```yaml
parallel_analysis:
  _type: parallel_executor
  tool_list: [branch_a, branch_b, branch_c]
  description: "Run multiple analyses in parallel"
```

### Composition with sequential_executor

The real power is in composition. You can nest these components:

```yaml
functions:
  # A sequential sub-pipeline
  nli_pipeline:
    _type: sequential_executor
    tool_list: [nli_agent, context_summary_agent]

  # Runs nli_pipeline and user_understanding in parallel
  parallel_analysis:
    _type: parallel_executor
    tool_list: [nli_pipeline, user_understanding_agent]

# Main workflow: Sequential(RAG → Parallel(Seq, Single) → Ranker)
workflow:
  _type: sequential_executor
  tool_list: [rag_retriever, parallel_analysis, item_ranker_agent]
```

## Configuration

The full configuration in `src/nat_arag/configs/config.yml`:

```yaml
functions:
  rag_retriever:
    _type: rag_retriever
    top_k: 5

  nli_agent:
    _type: nli_agent

  context_summary_agent:
    _type: context_summary_agent

  user_understanding_agent:
    _type: user_understanding_agent

  item_ranker_agent:
    _type: item_ranker_agent

  # Sequential sub-pipeline: NLI → Context Summary
  nli_pipeline:
    _type: sequential_executor
    tool_list: [nli_agent, context_summary_agent]
    raise_type_incompatibility: false

  # Parallel executor: fans out to both branches
  parallel_analysis:
    _type: parallel_executor
    tool_list: [nli_pipeline, user_understanding_agent]

workflow:
  _type: sequential_executor
  tool_list: [rag_retriever, parallel_analysis, item_ranker_agent]
  raise_type_incompatibility: false
```

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

From the root directory of the NeMo Agent Toolkit library, run the following command:

```bash
uv pip install -e examples/agentic_rag/arag
```

## Run the Workflow

Run the ARAG pipeline from the root of the NeMo Agent Toolkit repository:

```bash
nat run --config_file=examples/agentic_rag/arag/src/nat_arag/configs/config.yml --input "I need a good pair of headphones for working from home"
```

**Expected output:**

```
=== ARAG Personalized Recommendations ===

User Preferences: User is interested in technology products, specifically in categories: electronics. Price sensitivity: medium.
Context: Query: I need a good pair of headphones for working from home | Total candidates evaluated: 5 | Strong matches (entailment): 5 | Weak matches (neutral): 0 | Top aligned items: Wireless Noise-Cancelling Headphones, Smart Fitness Watch, Portable Bluetooth Speaker

Ranked Items:
  1. [0.553] Wireless Noise-Cancelling Headphones (NLI: entailment, Category: electronics)
  2. [0.503] Smart Fitness Watch (NLI: entailment, Category: electronics)
  3. [0.453] Portable Bluetooth Speaker (NLI: entailment, Category: electronics)
  4. [0.403] Mechanical Keyboard (NLI: entailment, Category: electronics)
  5. [0.350] Ergonomic Office Chair (NLI: entailment, Category: furniture)

=== End of Recommendations ===
```

The workflow demonstrates:
1. **Sequential execution**: RAG retrieval feeds into the parallel branches, which feed into the ranker
2. **Parallel execution**: The NLI pipeline and User Understanding Agent run concurrently
3. **Nested composition**: The NLI pipeline is itself a sequential sub-chain (NLI Agent → Context Summary)
