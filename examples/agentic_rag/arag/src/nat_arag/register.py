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
# ARAG Agent Functions (mock implementations for demonstration)
# =============================================================================


# -- 1. RAG Retriever --------------------------------------------------------


class RAGRetrieverConfig(FunctionBaseConfig, name="rag_retriever"):
    """Configuration for the RAG retriever function."""

    top_k: int = Field(default=10, description="Number of candidate items to retrieve.")


@register_function(config_type=RAGRetrieverConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rag_retriever_function(config: RAGRetrieverConfig, builder: Builder):
    """Retrieve an initial recall set of candidate items via RAG."""

    async def retrieve_candidates(user_query: str) -> str:
        """Simulate RAG retrieval of candidate items.

        In a real implementation this would query a vector store (e.g. Milvus)
        and return the top-k results.  Here we return mock data to illustrate
        the data flow.

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

        logger.info("RAG Retriever: retrieved %d candidates for query: %s", len(candidates), user_query)
        return json.dumps(retrieval_result)

    yield FunctionInfo.from_fn(retrieve_candidates, description="Retrieve candidate items using RAG")


# -- 2. NLI Agent ------------------------------------------------------------


class NLIAgentConfig(FunctionBaseConfig, name="nli_agent"):
    """Configuration for the NLI (Natural Language Inference) agent."""
    pass


@register_function(config_type=NLIAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def nli_agent_function(config: NLIAgentConfig, builder: Builder):
    """Evaluate semantic alignment between candidate items and inferred user intent."""

    async def evaluate_nli(retrieval_result: str) -> str:
        """Score each candidate item's alignment with the user's intent.

        In a real implementation this would call an LLM to perform NLI
        (entailment / contradiction / neutral) between user intent and each
        item's metadata.

        Args:
            retrieval_result: JSON string with user_query and candidates from RAG.

        Returns:
            JSON string with NLI-scored candidates.
        """
        data = json.loads(retrieval_result)
        user_query = data.get("user_query", "")
        candidates = data.get("candidates", [])

        query_lower = user_query.lower()
        scored_candidates = []
        for item in candidates:
            # Mock NLI scoring based on keyword overlap
            text = f"{item['title']} {item['description']} {item.get('reviews_summary', '')}".lower()
            query_words = set(query_lower.split())
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            nli_score = min(1.0, overlap / max(len(query_words), 1) * 0.8 + 0.2)

            scored_candidates.append({
                **item,
                "nli_score": round(nli_score, 3),
                "nli_label": "entailment" if nli_score > 0.6 else "neutral",
            })

        nli_result = {
            "user_query": user_query,
            "scored_candidates": scored_candidates,
        }

        logger.info("NLI Agent: scored %d candidates", len(scored_candidates))
        return json.dumps(nli_result)

    yield FunctionInfo.from_fn(evaluate_nli, description="Evaluate NLI alignment between candidates and user intent")


# -- 3. Context Summary Agent ------------------------------------------------


class ContextSummaryAgentConfig(FunctionBaseConfig, name="context_summary_agent"):
    """Configuration for the context summary agent."""
    pass


@register_function(config_type=ContextSummaryAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def context_summary_agent_function(config: ContextSummaryAgentConfig, builder: Builder):
    """Summarize the NLI findings into concise context for the ranker."""

    async def summarize_context(nli_result: str) -> str:
        """Produce a natural-language context summary from NLI-scored candidates.

        Args:
            nli_result: JSON string with NLI-scored candidates.

        Returns:
            JSON string containing the context summary and supporting items.
        """
        data = json.loads(nli_result)
        scored = data.get("scored_candidates", [])

        entailed = [c for c in scored if c.get("nli_label") == "entailment"]
        neutral = [c for c in scored if c.get("nli_label") == "neutral"]

        summary_lines = [
            f"Query: {data.get('user_query', 'N/A')}",
            f"Total candidates evaluated: {len(scored)}",
            f"Strong matches (entailment): {len(entailed)}",
            f"Weak matches (neutral): {len(neutral)}",
        ]

        if entailed:
            top_items = sorted(entailed, key=lambda x: x["nli_score"], reverse=True)[:3]
            summary_lines.append("Top aligned items: " + ", ".join(i["title"] for i in top_items))

        context_summary = {
            "summary": " | ".join(summary_lines),
            "entailed_items": entailed,
            "neutral_items": neutral,
        }

        logger.info("Context Summary Agent: %d entailed, %d neutral", len(entailed), len(neutral))
        return json.dumps(context_summary)

    yield FunctionInfo.from_fn(summarize_context, description="Summarize NLI findings into concise context")


# -- 4. User Understanding Agent ---------------------------------------------


class UserUnderstandingAgentConfig(FunctionBaseConfig, name="user_understanding_agent"):
    """Configuration for the user understanding agent."""
    pass


@register_function(config_type=UserUnderstandingAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def user_understanding_agent_function(config: UserUnderstandingAgentConfig, builder: Builder):
    """Generate a user preference summary from session and long-term context."""

    async def understand_user(retrieval_result: str) -> str:
        """Analyze the user's query and interaction context to infer preferences.

        In a real implementation this would look at the user's session history
        and long-term profile.  Here we simulate preference extraction.

        Args:
            retrieval_result: JSON string with user_query and candidates from RAG.

        Returns:
            JSON string with inferred user preferences.
        """
        data = json.loads(retrieval_result)
        user_query = data.get("user_query", "")

        # Mock user preference inference based on the query
        query_lower = user_query.lower()
        inferred_preferences = {
            "primary_interest": "technology" if any(
                kw in query_lower for kw in ["tech", "gadget", "electronic", "computer"]
            ) else "general",
            "price_sensitivity": "medium",
            "brand_affinity": "none detected",
            "preferred_categories": [],
        }

        # Infer categories from query keywords
        category_keywords = {
            "electronics": ["headphone", "speaker", "watch", "keyboard", "phone", "laptop", "tech", "gadget"],
            "furniture": ["chair", "desk", "office", "ergonomic"],
            "fitness": ["fitness", "exercise", "health", "workout", "sport"],
        }
        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                inferred_preferences["preferred_categories"].append(category)

        if not inferred_preferences["preferred_categories"]:
            inferred_preferences["preferred_categories"] = ["electronics"]

        user_profile = {
            "user_query": user_query,
            "session_preferences": inferred_preferences,
            "preference_summary": (
                f"User is interested in {inferred_preferences['primary_interest']} products, "
                f"specifically in categories: {', '.join(inferred_preferences['preferred_categories'])}. "
                f"Price sensitivity: {inferred_preferences['price_sensitivity']}."
            ),
        }

        logger.info("User Understanding Agent: inferred preferences for query: %s", user_query)
        return json.dumps(user_profile)

    yield FunctionInfo.from_fn(understand_user, description="Generate user preference summary from session context")


# -- 5. Item Ranker Agent -----------------------------------------------------


class ItemRankerAgentConfig(FunctionBaseConfig, name="item_ranker_agent"):
    """Configuration for the item ranker agent."""
    pass


@register_function(config_type=ItemRankerAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def item_ranker_agent_function(config: ItemRankerAgentConfig, builder: Builder):
    """Rank items based on combined NLI context and user understanding signals."""

    async def rank_items(parallel_results: str) -> str:
        """Produce a final ranked list by integrating context summary and user preferences.

        The input is the merged output from the parallel executor, which
        contains both the NLI pipeline context summary and the user
        understanding profile keyed by their tool names.

        Args:
            parallel_results: JSON string with merged parallel branch outputs.

        Returns:
            Formatted ranked recommendation list.
        """
        branches = json.loads(parallel_results)

        # Parse the NLI pipeline output (context summary)
        context_data = {}
        user_data = {}
        for key, value in branches.items():
            parsed = json.loads(value) if isinstance(value, str) else value
            if "entailed_items" in parsed or "summary" in parsed:
                context_data = parsed
            elif "session_preferences" in parsed or "preference_summary" in parsed:
                user_data = parsed

        # Combine signals to produce final ranking
        entailed_items = context_data.get("entailed_items", [])
        neutral_items = context_data.get("neutral_items", [])
        all_items = entailed_items + neutral_items

        preferred_categories = (
            user_data.get("session_preferences", {}).get("preferred_categories", [])
        )

        # Score items by combining NLI score + category preference bonus
        for item in all_items:
            nli_score = item.get("nli_score", 0.0)
            category_bonus = 0.15 if item.get("category") in preferred_categories else 0.0
            item["final_score"] = round(nli_score + category_bonus, 3)

        ranked = sorted(all_items, key=lambda x: x["final_score"], reverse=True)

        # Format the output
        report_lines = [
            "=== ARAG Personalized Recommendations ===",
            "",
            f"User Preferences: {user_data.get('preference_summary', 'N/A')}",
            f"Context: {context_data.get('summary', 'N/A')}",
            "",
            "Ranked Items:",
        ]
        for rank, item in enumerate(ranked, 1):
            report_lines.append(
                f"  {rank}. [{item['final_score']:.3f}] {item['title']} "
                f"(NLI: {item.get('nli_label', 'N/A')}, Category: {item.get('category', 'N/A')})"
            )

        report_lines.extend(["", "=== End of Recommendations ==="])

        logger.info("Item Ranker Agent: produced ranking of %d items", len(ranked))
        return "\n".join(report_lines)

    yield FunctionInfo.from_fn(rank_items, description="Rank items using combined NLI context and user preferences")
