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

from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .snippets import summary_instructions


class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="Names of entities that weren't extracted")


class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='UUID of the entity')
    name: str = Field(description='Name of the entity')
    entity_type: str | None = Field(
        default=None,
        description='Type of the entity. Must be one of the provided types or None',
    )


class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ..., description='List of entities classification triples.'
    )


class EntitySummary(BaseModel):
    summary: str = Field(
        ...,
        description=f'Summary containing the important information about the entity. Under {MAX_SUMMARY_CHARS} characters.',
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    reflexion: PromptVersion
    classify_nodes: PromptVersion
    extract_attributes: PromptVersion
    extract_summary: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    reflexion: PromptFunction
    classify_nodes: PromptFunction
    extract_attributes: PromptFunction
    extract_summary: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from conversational messages. 
    Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation."""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

Instructions:

You are given a conversation context and a CURRENT MESSAGE. Your task is to extract **entity nodes** mentioned **explicitly or implicitly** in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the 
reference entities. Only extract distinct entities from the CURRENT MESSAGE. Don't extract pronouns like you, me, he/she/they, we/us as entities.

1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract all significant entities, concepts, or actors that are **explicitly or implicitly** mentioned in the CURRENT MESSAGE.
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only).

3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate `entity_type_id` for each one.

4. **Exclusions**:
   - Do NOT extract entities representing relationships or actions.
   - Do NOT extract dates, times, or other temporal information—these will be handled separately.

5. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available).

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from JSON. 
    Your primary task is to extract and classify relevant entities from JSON files"""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<SOURCE DESCRIPTION>:
{context['source_description']}
</SOURCE DESCRIPTION>
<JSON>
{context['episode_content']}
</JSON>

{context['custom_prompt']}

Given the above source description and JSON, extract relevant entities from the provided JSON.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

Guidelines:
1. Extract all entities that the JSON represents. This will often be something like a "name" or "user" field
2. Extract all entities mentioned in all other properties throughout the JSON structure
3. Do NOT extract any properties that contain dates
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from an act, usually contains ~50 lines of dialogues.
    Your primary task is to extract and classify the speaker and other significant entities mentioned in the provided text."""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<TEXT>
{context['episode_content']}
</TEXT>

Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

{context['custom_prompt']}

Guidelines:
1. **Person Extraction**:
   - **Canonicalization (Narrator/Self)**:
     - The First Person ('我') and the Narrator ('旁白') MUST always be extracted as the named node **'Aria'**.
     - NEVER create nodes named '我', '旁白'.
   
   - **Noise Filtering (Generic Roles)**:
     - DO NOT extract generic, unnamed background characters (e.g., '路人甲', '工作人员').
     - **Exception**: Only extract them if they are CRITICAL to the act AND appear in multiple interactions.

   - **Name Resolution **:
     - **Strict Rule**: Prefer to use the first name.
     - **Action**: 
       - If the name is revealed (e.g., "The stranger said, 'Call me **Dante**'"), extract as 'Dante'.
2. **RelationshipView Extraction**:
    - **Format**: Must strictly use 'holder_name:target_name'.
    - **Name Resolution**: 
        - Use the **First Name** of the person. 
        - **EXCEPTION**: If there are multiple people with the same first name (e.g., two 'John's), you MUST include the Last Name or a distinct descriptor to avoid collision (e.g., 'JohnS', 'JohnD').
    - **Type Constraint**: Both Holder and Target must be strictly of type **Person**. Do not extract for places, organizations, or objects.
    - **Directionality (CRITICAL)**: 
        - The **Holder** is the person *feeling, thinking, or perceiving* of another person.
        - The **Target** is the person *being observed*.
        - Example: If text says "Paris finds Elliot annoying", extract 'Paris:Elliot' (Paris holds the view).
    - **Extraction Threshold**:
        - **Always Extract**: 'Aria:Paris', 'Aria:Elliot' (if they interact in the scene).
        - **Others**: Only extract a RelationshipView if there is a **meaningful interaction** involving:
            1. An explicit opinion or sentiment (e.g., liking, distrusting).
            2. A change in relationship status (e.g., meeting for the first time, fighting).
            3. If the act only has Aria and one other person, extract the RelationshipView of Aria to the other person.

3. **Preference Extraction**:
   - **Scope Restriction (Non-Human Only)**:
     - STRICTLY RESTRICT preferences to **inanimate objects, concepts, locations, or activities** (e.g., Food, Music, Hobbies, Weather, Color, etc.).
     - **FORBIDDEN CATEGORY**: The `category` field MUST NOT be 'Person', 'Character', 'Human', or 'Individual'.
   
   - **Differentiation Rule (Crucial)**:
     - If the text indicates liking/disliking a **specific person** (e.g., "I like Paris"), this is a **RelationshipView**, not a Preference. Do NOT extract it here.
     - If the text indicates liking a **type/role** of person generally (e.g., "I like honest people"), this IS a Preference (preference_category: 'Personality Trait', preference_value: 'Honesty').

   - **Formatting**:
     - Ensure the Preference has a clear Category and Value.
     - *Valid Example*: Category: 'Drink', Value: 'Black Coffee'.
     - *Invalid Example*: Category: 'Person', Value: 'Alice'. (STOP -> Move to RelationshipView).
4. **MemoryNote Extraction**:
   - **Quantity Control (Strict)**: 
     - You MUST extract **at least 1** MemoryNote.
     - You must NOT extract more than **2** MemoryNotes.
   
   - **Content Hierarchy**:
     - **Note #1 (The Summary)**: Extract the primary topic, event, or action of this conversation. If the conversation is trivial, summarize the casual topic (e.g., "我和Paris第一次相遇了").
     - **Note #2 (The Insight - Optional)**: Only extract a second note if there is a distinct, separate critical piece of information (e.g., a specific promise made, a secret revealed, or a plan changed,etc.).

   - **Writing Style**:
     - **Concise**: Keep each note under 20 words.
     - **Objective**: Use third-person past tense (e.g., "Aria expressed concern about the mission," NOT "I am worried").
     - **Self-Contained**: The note should make sense on its own without reading the chat history.

5. Extract significant entities, concepts, or actors mentioned in the act.
6. Avoid creating nodes for locations.
7. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).
8. Be as explicit as possible in your node names, using first names and avoiding abbreviations.
9. Please output everything in Chinese,but do not translate the Person name in English.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which entities have not been extracted from the given context"""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context['extracted_entities']}
</EXTRACTED ENTITIES>

Given the above previous messages, current message, and list of extracted entities; determine if any entities haven't been
extracted.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies entity nodes given the context from which they were extracted"""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
    {to_prompt_json([ep for ep in context['previous_episodes']])}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context['episode_content']}
    </CURRENT MESSAGE>

    <EXTRACTED ENTITIES>
    {context['extracted_entities']}
    </EXTRACTED ENTITIES>

    <ENTITY TYPES>
    {context['entity_types']}
    </ENTITY TYPES>

    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted entities.

    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the following ENTITY, update any of its attributes based on the information provided
        in MESSAGES. Use the provided attribute descriptions to better understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate entity property values if they cannot be found in the current context.
        2. Only use the provided MESSAGES and ENTITY to set attribute values.
        3. Resolve generic names: If a generic reference (e.g., '陌生男子') corresponds to a specific named entity identified in the text (e.g., 'Paris'), replace the generic name with the specific name in the Person entity.
        4. Maintain consistency: Update the RelationshipView name to use the latest holde_name and target_name. For example, change 'Aria:陌生男子' to 'Aria:Paris' so it Paris is the latest target_name.
        

        <MESSAGES>
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


def extract_summary(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity summaries from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the ENTITY, update the summary that combines relevant information about the entity
        from the messages and relevant information from the existing summary.
        Please restrict the summary to be no more than 2 sentences.

        {summary_instructions}

        <MESSAGES>
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'reflexion': reflexion,
    'extract_summary': extract_summary,
    'classify_nodes': classify_nodes,
    'extract_attributes': extract_attributes,
}
