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
from nat.data_models.component_ref import LLMRef
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
# ARAG Agent Functions — each agent is backed by an LLM
# =============================================================================


# -- 1. RAG Retriever --------------------------------------------------------
# The retriever simulates a vector-store lookup (Milvus, FAISS, etc.).
# In production, replace with a real nat_retriever + embedder.


class RAGRetrieverConfig(FunctionBaseConfig, name="rag_retriever"):
    """Configuration for the RAG retriever function."""

    top_k: int = Field(default=10, description="Number of candidate items to retrieve.")


@register_function(config_type=RAGRetrieverConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rag_retriever_function(config: RAGRetrieverConfig, builder: Builder):
    """Retrieve an initial recall set of candidate items via RAG.

    Note: This uses mock data to demonstrate the pipeline data flow.
    In production, wire this to a real retriever (e.g. milvus_retriever).
    """

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


# -- 2. NLI Agent (LLM-backed) -----------------------------------------------


class NLIAgentConfig(FunctionBaseConfig, name="nli_agent"):
    """Configuration for the NLI (Natural Language Inference) agent."""

    llm_name: LLMRef = Field(description="The LLM to use for NLI evaluation.")


@register_function(config_type=NLIAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def nli_agent_function(config: NLIAgentConfig, builder: Builder):
    """Evaluate semantic alignment between candidate items and inferred user intent using an LLM."""

    from langchain_core.prompts.chat import ChatPromptTemplate

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    system_prompt = """\
You are a Natural Language Inference (NLI) agent for a recommendation system.

Given a user query and a list of candidate items, evaluate how well each item \
aligns with the user's intent. For each item, produce:
- "nli_label": one of "entailment" (strong match), "neutral" (partial match), \
or "contradiction" (poor match).
- "nli_score": a float between 0.0 and 1.0 indicating alignment strength.
- "reasoning": a brief explanation of why the item does or does not match.

Respond with ONLY a valid JSON object in this exact format:
{{
  "user_query": "<the original query>",
  "scored_candidates": [
    {{
      "id": "<item id>",
      "title": "<item title>",
      "description": "<item description>",
      "category": "<item category>",
      "reviews_summary": "<item reviews>",
      "nli_label": "entailment | neutral | contradiction",
      "nli_score": <0.0-1.0>,
      "reasoning": "<brief explanation>"
    }}
  ]
}}"""

    user_prompt = "{input}"

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    chain = prompt | llm

    async def evaluate_nli(retrieval_result: str) -> str:
        """Use an LLM to perform NLI scoring on each candidate item.

        Args:
            retrieval_result: JSON string with user_query and candidates from RAG.

        Returns:
            JSON string with NLI-scored candidates.
        """
        response = await chain.ainvoke({"input": retrieval_result})
        logger.info("NLI Agent: completed LLM-based evaluation")
        return response.text()

    yield FunctionInfo.from_fn(evaluate_nli, description="Evaluate NLI alignment between candidates and user intent")


# -- 3. Context Summary Agent (LLM-backed) -----------------------------------


class ContextSummaryAgentConfig(FunctionBaseConfig, name="context_summary_agent"):
    """Configuration for the context summary agent."""

    llm_name: LLMRef = Field(description="The LLM to use for context summarization.")


@register_function(config_type=ContextSummaryAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def context_summary_agent_function(config: ContextSummaryAgentConfig, builder: Builder):
    """Summarize NLI findings into concise context for the ranker using an LLM."""

    from langchain_core.prompts.chat import ChatPromptTemplate

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    system_prompt = """\
You are a Context Summary agent for a recommendation system.

You receive the output of an NLI (Natural Language Inference) evaluation that \
scored candidate items against a user's query. Your job is to produce a concise \
natural-language summary that a downstream Item Ranker agent can use.

Your summary must include:
1. The original user query.
2. How many candidates were evaluated and the breakdown by NLI label \
(entailment / neutral / contradiction).
3. The key themes or attributes that made top items align with the query.
4. Any notable mismatches or gaps in the candidate set.

Respond with ONLY a valid JSON object in this exact format:
{{
  "summary": "<your natural language summary paragraph>",
  "top_aligned_items": ["<title1>", "<title2>", ...],
  "key_themes": ["<theme1>", "<theme2>", ...],
  "gaps": "<any gaps or missing categories noted>"
}}"""

    user_prompt = "{input}"

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    chain = prompt | llm

    async def summarize_context(nli_result: str) -> str:
        """Use an LLM to summarize NLI-scored candidates into context.

        Args:
            nli_result: JSON string with NLI-scored candidates.

        Returns:
            JSON string containing the context summary.
        """
        response = await chain.ainvoke({"input": nli_result})
        logger.info("Context Summary Agent: completed LLM-based summarization")
        return response.text()

    yield FunctionInfo.from_fn(summarize_context, description="Summarize NLI findings into concise context")


# -- 4. User Understanding Agent (LLM-backed) --------------------------------


class UserUnderstandingAgentConfig(FunctionBaseConfig, name="user_understanding_agent"):
    """Configuration for the user understanding agent."""

    llm_name: LLMRef = Field(description="The LLM to use for user preference analysis.")


@register_function(config_type=UserUnderstandingAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def user_understanding_agent_function(config: UserUnderstandingAgentConfig, builder: Builder):
    """Generate a user preference summary from session and long-term context using an LLM."""

    from langchain_core.prompts.chat import ChatPromptTemplate

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    system_prompt = """\
You are a User Understanding agent for a personalized recommendation system.

Given a user's query and the set of candidate items retrieved by RAG, infer \
the user's preferences, interests, and intent. Think about:
- What product categories the user cares about.
- What attributes matter most (price, quality, brand, features, etc.).
- Whether the query implies short-term (session) needs or long-term interests.
- Any implicit constraints (budget, use-case, lifestyle).

Respond with ONLY a valid JSON object in this exact format:
{{
  "user_query": "<the original query>",
  "preference_summary": "<a natural language paragraph summarizing the user's preferences>",
  "inferred_interests": {{
    "primary_category": "<main product category>",
    "key_attributes": ["<attr1>", "<attr2>", ...],
    "use_case": "<inferred use case>",
    "price_sensitivity": "low | medium | high",
    "session_vs_longterm": "session | longterm | both"
  }}
}}"""

    user_prompt = "{input}"

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    chain = prompt | llm

    async def understand_user(retrieval_result: str) -> str:
        """Use an LLM to analyze user intent and infer preferences.

        Args:
            retrieval_result: JSON string with user_query and candidates from RAG.

        Returns:
            JSON string with inferred user preferences.
        """
        response = await chain.ainvoke({"input": retrieval_result})
        logger.info("User Understanding Agent: completed LLM-based preference analysis")
        return response.text()

    yield FunctionInfo.from_fn(understand_user, description="Generate user preference summary from session context")


# -- 5. Item Ranker Agent (LLM-backed) ---------------------------------------


class ItemRankerAgentConfig(FunctionBaseConfig, name="item_ranker_agent"):
    """Configuration for the item ranker agent."""

    llm_name: LLMRef = Field(description="The LLM to use for final item ranking.")


@register_function(config_type=ItemRankerAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def item_ranker_agent_function(config: ItemRankerAgentConfig, builder: Builder):
    """Rank items based on combined NLI context and user understanding signals using an LLM."""

    from langchain_core.prompts.chat import ChatPromptTemplate

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    system_prompt = """\
You are an Item Ranker agent for a personalized recommendation system.

You receive two inputs merged into a single JSON object with two keys:
1. **NLI pipeline output** — a context summary of how well candidate items \
align with the user's query (includes NLI labels, scores, and themes).
2. **User understanding output** — an analysis of the user's preferences, \
interests, and intent.

Your task is to combine both signals and produce a final **ranked list** of \
recommended items, ordered from most relevant to least relevant.

For each item, explain briefly why it was ranked at that position, referencing \
both the NLI alignment and the user's preferences.

Respond with ONLY a valid JSON object in this exact format:
{{
  "ranked_recommendations": [
    {{
      "rank": 1,
      "title": "<item title>",
      "category": "<item category>",
      "relevance_score": <0.0-1.0>,
      "reasoning": "<why this item is ranked here, referencing NLI + user prefs>"
    }}
  ],
  "summary": "<a brief paragraph summarizing the overall recommendation rationale>"
}}"""

    user_prompt = "{input}"

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    chain = prompt | llm

    async def rank_items(parallel_results: str) -> str:
        """Use an LLM to produce a final ranked recommendation list.

        The input is the merged output from the parallel executor containing
        both the NLI pipeline context summary and the user understanding profile.

        Args:
            parallel_results: JSON string with merged parallel branch outputs.

        Returns:
            JSON string with ranked recommendations.
        """
        response = await chain.ainvoke({"input": parallel_results})
        logger.info("Item Ranker Agent: completed LLM-based ranking")
        return response.text()

    yield FunctionInfo.from_fn(rank_items, description="Rank items using combined NLI context and user preferences")
