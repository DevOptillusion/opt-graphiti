from math import factorial
from token import OP
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pytz import timezone as pytz_timezone

# Base Classes for Node and Edge
class BaseNode(BaseModel):
    """Base class for all node types in the graph."""
    # node_id: str = Field(..., description="Unique identifier for the node")
    # updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        arbitrary_types_allowed = True

class BaseEdge(BaseModel):
    """Base class for all edge/relationship types in the graph."""
    # edge_id: str = Field(..., description="Unique identifier for the edge")
    # source_node_id: str = Field(..., description="ID of the source node")
    # target_node_id: str = Field(..., description="ID of the target node")
    # source_episode_id: str = Field(..., description="ID of the source   ")
    # last_updated_episode_id: str = Field(..., description="ID of the last updated episode")
    
    class Config:
        arbitrary_types_allowed = True

# Custom Entity Types (Nodes)
class Person(BaseNode):
    """A person entity with biographical information."""
    person_name: str = Field(..., description="Name of the person,who should be important characters. Random Person with name like '路人甲','工作人员' should not be included.")
    age: Optional[int] = Field(None, description="Age of the person")
    occupation: Optional[str] = Field(None, description="Current occupation")
    birth_date: Optional[datetime] = Field(None, description="Date of birth")

# class SelfStat(BaseNode):
#     """A self-stated status of a person."""
#     mood: Optional[str] = Field(None, description="Mood of the person")
#     tone: Optional[str] = Field(None, description="Tone of the person")
#     energy: Optional[int] = Field(None, description="Energy level of the person")
#     social_battery: Optional[int] = Field(None, description="Social battery level of the person")
#     stress: Optional[int] = Field(None, description="Stress level of the person")
#     general_personality: Optional[str] = Field(None, description="General personality of the person")
#     focus: Optional[str] = Field(None, description="Focus of the person")
#     current_context_bias: Optional[str] = Field(None, description="Current context bias of the person")

# class Trait(BaseNode):
#     """A trait that describe a person's personality."""
#     # tricky, each trait need specific definition
#     trait_name: str = Field(..., description="Name of the trait, choose one from following:'Ambition','Confidence','Humor','Jealousy'")
#     decay_rate: Optional[float] = Field(None, description="Decay rate of the trait")
    

class Preference(BaseNode):
    """A preference that describe a character likes something. cannot be a person."""
    # need define
    perference_category: str = Field(..., description="Category of the preference, for example,'color',can not be 'person'. Note 'Paris' is a person, not a location.")
    preference_value: str = Field(..., description="Value of the preference, for example,'red' if preference category is 'color'.Note 'Paris' is a person, not a location.")
    evidence:str = Field(..., description="Verbatim excerpt from the act that serves as evidence")
    preference_strength:str = Field(..., description="""
    Assign strength based on how strongly the character expresses their preference. Use this rubric:

    low: Weak or casual preference. Words like "还行", "可以","不错","偶尔喜欢","不讨厌".

    medium: Clear positive or negative preference, but not extreme. Words like "喜欢", "讨厌", "偏爱", "经常选择".

    high: Very strong or intense expression. Words like "最喜欢", "超级喜欢", "非常讨厌", "离不开", "特别重要".

    If no explicit preference strength is expressed, default to medium.
    Output strength label, e.g. "medium".
    """)
 
class Belief(BaseNode):
    """A belief held by a person. not shared with other people"""
    # need restricted belief type  about a relationship between two person
    holder_name: str = Field(..., description="name of the holder person, once is set, cannot be changed, when updating, keep the same holder_name")
    belief_content: str = Field(..., description="It refers to a character’s(holder) subjective perception of a fact, event, interpersonal relationship, or the state of the world.")
    evidence_spans: Optional[List[str]] = Field(None, description="""
        A List of evidence spans
        Each span should be a short verbatim excerpt (1–3 lines) that directly supports the belief_content under evaluation. 
        Keep all previous evidence in the spans when updating.
    """)
    confidence: str = Field(None, description="""
        Confidence(p)defines how strong the holder believes the belief content.
        *to calculate confidence*
        Label each span as supports or contradicts the belief in evidence_spans.
        For each span, set weight ∈ {1,2}: 2 if the span is a direct/explicit statement or decisive action; otherwise 1.
        Let S = sum(weight for support), C = sum(weight for contradict), total = S+C. If total=0, output confidence 0.5.
        Compute Confidence(p) = S/total. Snap p to confidence bins:
        p ≤ 0.10 → 0.00; 
        0.10 < p ≤ 0.35 → 0.25; 
        0.35 < p < 0.65 → 0.50; 
        0.65 ≤ p < 0.90 → 0.75; 
        p ≥ 0.90 → 1.00.
    """)

class RelationshipView(BaseNode):
    """A relationshipview represent a person's view of a another person"""
    holder_name: str = Field(..., description="name of the holder person who hold the relationship view.Prefer using First Name.")
    target_name: str = Field(..., description="name of the target person, that the holder person has a relationship view towards. Prefer using First Name.")

    shared_history: Optional[List[str]] = Field(None, description="""
        shared_history is a list of shared events between the holder and the target.
        Each item is a key event summary of 1-2 sentences (concise, no spoilers, in Chinese) that the holder and the target have shared.
        Prefer events that: (1) involve both characters directly, (2) reveal personal info or trust, (3) show repeated interaction or routines.
        Do not list generic crowd scenes or one-off trivialities.
        **Update rule**
        When a new qualifying joint event occurs, append and keep the previous shared_history
    """)
    #nickname
    familarity: str = Field(None, description="""
        Assign familiarity ∈ {Stranger,Acquaintance,Familiar,Close,Intimate} .Do not confuse familiarity with affection. Even enemies can have high familiarity if they share a long or deep history.
        
        **To calculate familiarity**
        1.Count how many shared_history events there are (count).
        2.Assign a salience value to each shared_history event based on the description below.
        salience ∈ {0.3, 0.5, 0.7, 0.9} based on how much this event increases familiarity.Tiny anchors for salience:
            0.3 minor shared interaction (brief talk, first introduction);
            0.5 meaningful conversation or help;
            0.7 significant personal disclosure / living together routine;
            0.9 transformational event (confession, betrayal, rescue).
        3.Sum their salience values (Σsalience).
        4.Compute score = (0.7 * Σsalience + 0.3 * count).
        5.Familiarity = 1 - exp(-score) (cap at 0.9).

        **Map result to {Stranger,Acquaintance,Familiar,Close,Intimate}**
        0.0–0.2 → Stranger
        0.3–0.4 → Acquaintance
        0.5–0.6 → Familiar
        0.7–0.8 → Close
        0.9 → Intimate
        Default to Stranger if there is no shared history.
    """) 

    current_attitude: str = Field(None, description="""
        Assign current_attitude∈ {"hostile", "cold", "neutral", "warm", "affectionate"} to represent the holder’s immediate emotional stance toward the target in the current act.
        Focus only on the latest interactions, not long history.
        Do not mix the current attitude with familarity.
        **Use this scale:**
        "hostile" (≈ -0.8): insults, rejection, aggression
        "cold" (≈ -0.4): dismissive, indifferent, distant
        "neutral" (0.0): polite, no clear emotion
        "warm" (≈ +0.4): supportive, caring, friendly
        "affectionate" (≈ +0.8): loving, protective, deeply caring
        If mixed signals, average them; lean toward the strongest most recent evidence.
    """)
    current_attitude_evidence_summary: str = Field(None, description="summary of the evidence in the current act that supports the current attitude. Kepp it in 1-3 sentences in Chinese.")
   
class MemoryNote(BaseNode):
    """A key memory note from an episode that a person remembers."""
    memory_summary: str = Field(..., description="A key memory event from an episode that a person remembers. Restrict to 3 sentences in Chinese.")
    involved_persons: List[str] = Field(..., description="List of persons involved in the memory event. Restrict to at most 3 persons, based on importance.")

# class EpisodeSummary(BaseNode):
#     """A summary of an  episode."""
#     content: str = Field(..., description="Summary content of an episode")
#     episode_id: [int] = Field(None, description="ID of the episode")
    

# Custom Edge Types

# class HasTrait(BaseEdge):
#    """Edge representing a person having a trait."""


class HasPreference(BaseEdge):
    """Edge representing a person having a preference."""


class HasBelief(BaseEdge):
    """Edge representing a person holding a belief."""

class HasRelationshipView(BaseEdge):
    """Edge representing a person having a relationship view towards another person."""

class HasMemory(BaseEdge):
    """Edge representing a person having a memory."""

class HasEpisodeSummary(BaseEdge):
    """Edge representing that a node has an associated episode summary.
    
    Can connect:
    - Person -> EpisodeSummary (person has an episode summary)
    """

class About(BaseEdge):
    """Edge representing that a node is about another node.
    
    Can connect:
    - MemoryNote -> Person (memory is about a person)
    - Belief -> Person (belief is about a person)
    - RelationshipView -> Person (relationship view is about a target person)
    """

# Entity type registry
entity_types = {
    "Person": Person,
    "Preference": Preference,
    "RelationshipView": RelationshipView,
    "Belief": Belief,   
    "MemoryNote": MemoryNote, 
}

# Edge types registry
edge_types = {
    "HasPreference": HasPreference,
    "HasRelationshipView": HasRelationshipView,
    "HasBelief": HasBelief,
    "HasMemory": HasMemory,
    "About": About,
}



# Edge Type Map:
edge_type_map = {
    ("Person","Preference"): ["HasPreference"],
    ("Person","RelationshipView"): ["HasRelationshipView"],
    ("Person","Belief"): ["HasBelief"],
    ("Person","MemoryNote"): ["HasMemory"],
    ("MemoryNote","Person"): ["About"],
    ("Belief","Person"): ["About"],
    #("RelationshipView","Person"): ["About"],
}