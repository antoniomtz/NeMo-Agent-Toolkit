# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ARAG custom components.

Only two components require custom Python code:

1. **parallel_executor** — a new control-flow component that fans-out the
   same input to multiple tools, executes them concurrently via
   ``asyncio.gather``, and merges their outputs into a single JSON object.

2. **rag_retriever** — a mock vector-store retriever that returns sample
   candidate items.  In production, replace with a real ``nat_retriever``
   backed by Milvus or another vector database.

All four LLM-backed agents (NLI, Context Summary, User Understanding,
Item Ranker) are defined **entirely in config.yml** using the built-in
``chat_completion`` type — no Python code needed.
"""

import asyncio
import json
import logging

from langchain_core.tools.base import BaseTool
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Parallel Executor — a new control flow component for fan-out / fan-in
# =============================================================================


class ParallelExecutorConfig(FunctionBaseConfig, name="parallel_executor"):
    """Configuration for parallel execution of a list of functions.

    Every function in ``tool_list`` receives the **same** input (the output of
    the preceding step).  All functions execute concurrently via
    ``asyncio.gather``.  Their outputs are collected into a JSON object keyed
    by function name so the next step in the pipeline can consume them.
    """

    description: str = Field(
        default="Parallel Executor Workflow",
        description="Description of this function's use.",
    )
    tool_list: list[FunctionRef] = Field(
        default_factory=list,
        description="A list of functions to execute in parallel.",
    )


@register_function(config_type=ParallelExecutorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def parallel_execution(config: ParallelExecutorConfig, builder: Builder):
    """Create a parallel executor that fans-out input to all tools and merges their outputs."""

    tools: list[BaseTool] = await builder.get_tools(
        tool_names=config.tool_list,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )
    tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    async def _parallel_function_execution(input_message: str) -> str:
        """Execute all configured tools in parallel and merge results.

        Args:
            input_message: Input from the previous step (broadcast to every tool).

        Returns:
            JSON string mapping each tool name to its output.
        """
        logger.debug("Parallel executor: launching %d tools in parallel", len(config.tool_list))

        tasks = []
        tool_names: list[str] = []
        for tool_name in config.tool_list:
            tool = tools_dict[tool_name]
            tasks.append(tool.ainvoke(input_message))
            tool_names.append(str(tool_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: dict[str, str] = {}
        for name, result in zip(tool_names, results):
            if isinstance(result, BaseException):
                logger.error("Parallel executor: tool %s failed: %s", name, result)
                merged[name] = f"ERROR: {result}"
            else:
                merged[name] = str(result)

        logger.debug("Parallel executor: all tools completed")
        return json.dumps(merged)

    yield FunctionInfo.from_fn(_parallel_function_execution, description=config.description)


# =============================================================================
# RAG Retriever — mock vector-store retriever for demonstration
# =============================================================================


class RAGRetrieverConfig(FunctionBaseConfig, name="rag_retriever"):
    """Configuration for the RAG retriever function.

    In production, replace this with a real ``nat_retriever`` backed by
    Milvus, FAISS, or another vector database.
    """

    top_k: int = Field(default=10, description="Number of candidate items to retrieve.")


@register_function(config_type=RAGRetrieverConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rag_retriever_function(config: RAGRetrieverConfig, builder: Builder):
    """Retrieve an initial recall set of candidate items via RAG."""

    async def retrieve_candidates(user_query: str) -> str:
        """Simulate RAG retrieval of candidate items.

        Args:
            user_query: The user's recommendation request.

        Returns:
            JSON string containing the query and a list of candidate items.
        """
        candidates = [
            {
                "id": "item_001",
                "title": "Wireless Noise-Cancelling Headphones",
                "description": "Premium over-ear headphones with active noise cancellation and 30-hour battery life.",
                "category": "electronics",
                "reviews_summary": "Great sound quality, comfortable for long sessions.",
            },
            {
                "id": "item_002",
                "title": "Smart Fitness Watch",
                "description": "Advanced fitness tracker with heart rate monitoring, GPS, and sleep analysis.",
                "category": "electronics",
                "reviews_summary": "Accurate tracking, good battery life, sleek design.",
            },
            {
                "id": "item_003",
                "title": "Portable Bluetooth Speaker",
                "description": "Waterproof speaker with 360-degree sound and 20-hour battery life.",
                "category": "electronics",
                "reviews_summary": "Excellent sound for its size, truly waterproof.",
            },
            {
                "id": "item_004",
                "title": "Ergonomic Office Chair",
                "description": "Adjustable lumbar support, breathable mesh back, and memory foam seat.",
                "category": "furniture",
                "reviews_summary": "Very comfortable for all-day use, easy to assemble.",
            },
            {
                "id": "item_005",
                "title": "Mechanical Keyboard",
                "description": "RGB backlit mechanical keyboard with Cherry MX switches and USB-C.",
                "category": "electronics",
                "reviews_summary": "Satisfying key feel, great for both typing and gaming.",
            },
        ]

        retrieval_result = {
            "user_query": user_query,
            "candidates": candidates[:config.top_k],
        }

        logger.info("RAG Retriever: retrieved %d candidates for query: %s",
                     len(retrieval_result["candidates"]), user_query)
        return json.dumps(retrieval_result)

    yield FunctionInfo.from_fn(retrieve_candidates, description="Retrieve candidate items using RAG")
