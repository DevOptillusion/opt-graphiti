"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json


class NodeDuplicate(BaseModel):
    id: int = Field(..., description='integer id of the entity')
    duplicate_idx: int = Field(
        ...,
        description='idx of the duplicate entity. If no duplicate entities are found, default to -1.',
    )
    name: str = Field(
        ...,
        description='Name of the entity. Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}.',
    )
    duplicates: list[int] = Field(
        ...,
        description='idx of all entities that are a duplicate of the entity with the above id.',
    )


class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate] = Field(..., description='List of resolved nodes')


class Prompt(Protocol):
    node: PromptVersion
    node_list: PromptVersion
    nodes: PromptVersion


class Versions(TypedDict):
    node: PromptFunction
    node_list: PromptFunction
    nodes: PromptFunction


def node(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that determines whether or not a NEW ENTITY is a duplicate of any EXISTING ENTITIES.',
        ),
        Message(
            role='user',
            content=f"""
        <PREVIOUS MESSAGES>
        {to_prompt_json([ep for ep in context['previous_episodes']])}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context['episode_content']}
        </CURRENT MESSAGE>
        <NEW ENTITY>
        {to_prompt_json(context['extracted_node'])}
        </NEW ENTITY>
        <ENTITY TYPE DESCRIPTION>
        {to_prompt_json(context['entity_type_description'])}
        </ENTITY TYPE DESCRIPTION>

        <EXISTING ENTITIES>
        {to_prompt_json(context['existing_nodes'])}
        </EXISTING ENTITIES>
        
        Given the above EXISTING ENTITIES and their attributes, MESSAGE, and PREVIOUS MESSAGES; Determine if the NEW ENTITY extracted from the conversation
        is a duplicate entity of one of the EXISTING ENTITIES with the same entity_types.
        
        Entities should only be considered duplicates if they refer to the *same real-world object or concept*.
        Semantic Equivalence: if a descriptive label in existing_entities clearly refers to a named entity in context, treat them as duplicates.

        Guidelines for each entity_type:
        - Person: Be careful on Person, if you are not sure whether they are the same person, do not mark them as duplicates.
        - RelationshipView: it has format "holder_name:target_name", if you think two relationshipView Nodes have the same holder person and also the same target person, they are duplicates.
        - MemoryNote: Please never mark MemoryNote as duplicates to anothe MemoryNote Node.

        Do NOT mark entities as duplicates if:
        - They are related but distinct.
        - They have similar names or purposes but refer to separate instances or concepts.
        - They have different entity_types.

         TASK:
         1. Compare `new_entity` against each item in `existing_entities`.
         2. If it refers to the same real-world object or concept, collect its index.
         3. Let `duplicate_idx` = the smallest collected index, or -1 if none.
         4. Let `duplicates` = the sorted list of all collected indices (empty list if none).

        Respond with a JSON object containing an "entity_resolutions" array with a single entry:
        {{
            "entity_resolutions": [
                {{
                    "id": integer id from NEW ENTITY,
                    "name": the best full name for the entity,
                    "duplicate_idx": integer index of the best duplicate in EXISTING ENTITIES, or -1 if none,
                    "duplicates": sorted list of all duplicate indices you collected (deduplicate the list, use [] when none)
                }}
            ]
        }}

        Only reference indices that appear in EXISTING ENTITIES, and return [] / -1 when unsure.
        """,
        ),
    ]


def nodes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that determines whether or not ENTITIES extracted from a conversation are duplicates'
            ' of existing entities.',
        ),
        Message(
            role='user',
                content=f"""
<DATA_CONTEXT>
    <PREVIOUS MESSAGES>
    {to_prompt_json([ep for ep in context['previous_episodes'][-1:]])}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context['episode_content']}
    </CURRENT MESSAGE>
    
    <NEW ENTITIES>
    {to_prompt_json(context['extracted_nodes'])} 
    </NEW ENTITIES>
    
    <EXISTING ENTITIES>
    {to_prompt_json(context['existing_nodes'])}
    </EXISTING ENTITIES>
</DATA_CONTEXT>

<INSTRUCTIONS>
You have been provided with a list of `<NEW ENTITIES>`. 
**TASK:** Iterate through **EACH** item in `<NEW ENTITIES>` and determine if it is a duplicate of any item in `<EXISTING ENTITIES>`.

For **EVERY SINGLE ENTITY** in the list, execute this **Decision Algorithm**:

**STEP 1: CHECK THE ENTITY TYPE (Strict Logic Gate)**
Check the `entity_types` of the current entity being processed.

* **IF entity_types == "MemoryNote"**:
    * **RULE:** NEVER merge. MemoryNotes are distinct temporal thoughts.
    * **ACTION:** STOP. Return `duplicate_idx: -1`.

* **IF entity_types == "RelationshipView"**:
    * **RULE:** Strict Structural Match. The name format is "HolderName:TargetName". If you think the holder and target are the same person, they are duplicates.
    * **ACTION:** Only merge if you think the holder and target are the same person, they are duplicates. Note, sometimes the same target can have different names. For examples, a person was known as '陌生男人'，but based on the history, it can be the same target of a later one.

* **IF entity_types == "Person"**:
    * **RULE:** High Ambiguity Caution. If the name matches，it is a duplicate.
    * **ACTION:** Use context to confirm identity. If names are similar but distinct (e.g. "John S." vs "John D."), return `duplicate_idx: -1`.

* **ALL OTHER entity_types**:
    * **ACTION:** Proceed to Step 2.

**STEP 2: SEMANTIC COMPARISON (Only if Step 1 allowed it)**
* Compare the current entity against `<EXISTING ENTITIES>`.
* Entities are duplicates ONLY if they refer to the *same real-world object*.
* Use `<PREVIOUS MESSAGES>` to resolve pronouns.

**STEP 3: OUTPUT**
* Add the result to the output array.
</INSTRUCTIONS>

Respond with a JSON object containing an "entity_resolutions" array. 
**The array must contain exactly one result object for every item in `<NEW ENTITIES>`**.

Structure:
{{
    "entity_resolutions": [
        {{
            "id": integer id from the NEW ENTITY being processed,
            "name": "the best full name",
            "duplicate_idx": integer index of the best duplicate in EXISTING ENTITIES, or -1 if none,
            "duplicates": [sorted list of all duplicate indices]
        }},
        ... (repeat for next entity) ...
    ]
}}

    - Only use idx values that appear in EXISTING ENTITIES.
    - Set duplicate_idx to the smallest idx you collected for that entity, or -1 if duplicates is empty.
    - Never fabricate entities or indices.
""",
        ),
    ]


def node_list(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates nodes from node lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, deduplicate a list of nodes:

        Nodes:
        {to_prompt_json(context['nodes'])}

        Task:
        1. Group nodes together such that all duplicate nodes are in the same list of uuids
        2. All duplicate uuids should be grouped together in the same list
        3. Also return a new summary that synthesizes the summary into a new short summary

        Guidelines:
        1. Each uuid from the list of nodes should appear EXACTLY once in your response
        2. If a node has no duplicates, it should appear in the response in a list of only one uuid

        Respond with a JSON object in the following format:
        {{
            "nodes": [
                {{
                    "uuids": ["5d643020624c42fa9de13f97b1b3fa39", "node that is a duplicate of 5d643020624c42fa9de13f97b1b3fa39"],
                    "summary": "Brief summary of the node summaries that appear in the list of names."
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {'node': node, 'node_list': node_list, 'nodes': nodes}
