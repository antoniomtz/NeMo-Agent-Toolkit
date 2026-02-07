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

import pytest


@pytest.mark.parametrize(
    "question, expected_answer",
    [
        # Test that the full ARAG pipeline produces ranked recommendations
        (
            "I need a good pair of headphones for working from home",
            "Ranked Items",
        ),
        (
            "Recommend tech gadgets for my home office",
            "Personalized Recommendations",
        ),
        (
            "I want a fitness tracker for daily exercise",
            "Ranked Items",
        ),
    ],
)
@pytest.mark.integration
async def test_arag_workflow(question: str, expected_answer: str) -> None:
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_arag.register import RAGRetrieverConfig

    config_file = locate_example_config(RAGRetrieverConfig)
    await run_workflow(config_file=config_file, question=question, expected_answer=expected_answer)
