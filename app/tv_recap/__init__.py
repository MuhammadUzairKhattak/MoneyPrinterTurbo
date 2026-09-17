"""TV Recap Engine for AI-generated TV episode recap videos.

This package adds a shot-based video generation pipeline alongside MPT's
existing script-to-footage workflow. It reuses MPT's voice, subtitle, music,
and FFmpeg assembly services while adding episode analysis, character
consistency, and structured shot planning.
"""

from app.tv_recap.models import (
    CharacterDescription,
    EpisodeInput,
    LocationDescription,
    RecapScript,
    Shot,
    ShowConfig,
)

__all__ = [
    "CharacterDescription",
    "EpisodeInput",
    "LocationDescription",
    "RecapScript",
    "Shot",
    "ShowConfig",
]
