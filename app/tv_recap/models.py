"""Data models for the TV Recap Engine.

All structured data flowing through the recap pipeline is defined here:
episodes, characters, locations, shots, and recap scripts.  Models use
Pydantic for validation so they can later be exposed as API request/response
schemas without a separate definition layer.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CameraAngle(str, Enum):
    """Common camera angles for shot planning."""
    wide = "wide"
    medium = "medium"
    close_up = "close-up"
    extreme_close_up = "extreme-close-up"
    over_the_shoulder = "over-the-shoulder"
    two_shot = "two-shot"
    tracking = "tracking"
    aerial = "aerial"
    pov = "pov"


class RecapState(str, Enum):
    """Pipeline execution states for a recap job."""
    pending = "pending"
    analyzing = "analyzing"
    planning = "planning"
    generating_shots = "generating_shots"
    assembling = "assembling"
    complete = "complete"
    failed = "failed"


# ---------------------------------------------------------------------------
# Character & Location
# ---------------------------------------------------------------------------

class CharacterDescription(BaseModel):
    """A single character in the show's character bible."""
    name: str
    gender: str = ""
    age_range: str = ""                     # e.g. "50s", "20s"
    appearance: str = ""                    # free-text physical description
    clothing: str = ""                      # default outfit description
    personality_traits: list[str] = Field(default_factory=list)
    reference_image: Optional[str] = None   # path to reference image

    def to_prompt_description(self) -> str:
        """Build a concise text description for video-generation prompts."""
        parts = [self.name]
        if self.gender:
            parts.append(self.gender)
        if self.age_range:
            parts.append(f"in their {self.age_range}")
        if self.appearance:
            parts.append(self.appearance)
        if self.clothing:
            parts.append(f"wearing {self.clothing}")
        return ", ".join(parts)


class LocationDescription(BaseModel):
    """A recurring location in the show."""
    name: str
    description: str = ""                   # free-text visual description
    reference_image: Optional[str] = None   # path to reference image

    def to_prompt_description(self) -> str:
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Show & Episode
# ---------------------------------------------------------------------------

class ShowConfig(BaseModel):
    """Per-show configuration stored on disk."""
    show_name: str
    style: str = ""                         # e.g. "dark crime drama", "comedy"
    tone: str = ""                          # e.g. "tense", "lighthearted"
    characters: list[CharacterDescription] = Field(default_factory=list)
    locations: list[LocationDescription] = Field(default_factory=list)

    def get_character(self, name: str) -> CharacterDescription | None:
        """Case-insensitive character lookup."""
        name_lower = name.lower()
        for char in self.characters:
            if char.name.lower() == name_lower:
                return char
        return None

    def get_location(self, name: str) -> LocationDescription | None:
        """Case-insensitive location lookup."""
        name_lower = name.lower()
        for loc in self.locations:
            if loc.name.lower() == name_lower:
                return loc
        return None


class EpisodeInput(BaseModel):
    """Raw input describing a single episode to recap."""
    show_name: str
    season: int = 1
    episode: int = 1
    title: str = ""
    summary: str                            # episode plot summary text
    language: str = "en"

    @property
    def episode_id(self) -> str:
        return f"S{self.season:02d}E{self.episode:02d}"


# ---------------------------------------------------------------------------
# Shot Plan
# ---------------------------------------------------------------------------

class Shot(BaseModel):
    """A single shot in the recap video."""
    index: int
    duration: int = 5                       # target duration in seconds
    characters: list[str] = Field(default_factory=list)  # character names
    location: str = ""
    action: str = ""                        # what happens in this shot
    camera: str = "medium"                  # camera angle
    prompt: str = ""                        # full video-generation prompt
    reference_images: list[str] = Field(default_factory=list)
    clip_path: Optional[str] = None         # filled after generation


class RecapScript(BaseModel):
    """The structured output of episode analysis + scene planning."""
    episode_id: str
    show_name: str
    total_duration: int = 60                # target total in seconds
    narration: str = ""                     # continuous narration text
    shots: list[Shot] = Field(default_factory=list)

    @property
    def planned_duration(self) -> int:
        """Sum of all shot durations."""
        return sum(shot.duration for shot in self.shots)


# ---------------------------------------------------------------------------
# Pipeline Job
# ---------------------------------------------------------------------------

class RecapJob(BaseModel):
    """Tracks the full state of a recap generation job."""
    task_id: str
    episode: EpisodeInput
    show_config: Optional[ShowConfig] = None
    recap_script: Optional[RecapScript] = None
    state: RecapState = RecapState.pending
    progress: int = 0
    error: Optional[str] = None

    # Paths filled during pipeline execution
    narration_audio: Optional[str] = None
    subtitle_file: Optional[str] = None
    final_video: Optional[str] = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str | dict) -> "RecapJob":
        if isinstance(data, str):
            data = json.loads(data)
        return cls.model_validate(data)
