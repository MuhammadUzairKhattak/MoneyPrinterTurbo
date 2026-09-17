"""Character bible management for TV Recap Engine.

Handles loading, saving, and querying per-show character bibles from disk.
Character bibles are stored as JSON files alongside reference images in the
``storage/shows/<show_name>/`` directory structure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from app.tv_recap.models import (
    CharacterDescription,
    LocationDescription,
    ShowConfig,
)
from app.utils import utils


def _shows_dir() -> str:
    """Root directory for all show data."""
    return utils.storage_dir("shows", create=True)


def _show_dir(show_name: str) -> str:
    """Directory for a specific show, creating it if needed."""
    safe_name = show_name.strip().replace(" ", "_").lower()
    d = os.path.join(_shows_dir(), safe_name)
    os.makedirs(d, exist_ok=True)
    return d


def _characters_dir(show_name: str) -> str:
    d = os.path.join(_show_dir(show_name), "characters")
    os.makedirs(d, exist_ok=True)
    return d


def _locations_dir(show_name: str) -> str:
    d = os.path.join(_show_dir(show_name), "locations")
    os.makedirs(d, exist_ok=True)
    return d


def _episodes_dir(show_name: str) -> str:
    d = os.path.join(_show_dir(show_name), "episodes")
    os.makedirs(d, exist_ok=True)
    return d


def _config_path(show_name: str) -> str:
    return os.path.join(_show_dir(show_name), "show_config.json")


# ---------------------------------------------------------------------------
# Show Config CRUD
# ---------------------------------------------------------------------------

def load_show_config(show_name: str) -> ShowConfig | None:
    """Load a show config from disk.  Returns None if not found."""
    config_file = _config_path(show_name)
    if not os.path.isfile(config_file):
        return None
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ShowConfig.model_validate(data)
    except Exception as exc:
        logger.warning(f"failed to load show config: {show_name}, error: {exc}")
        return None


def save_show_config(config: ShowConfig) -> str:
    """Persist a show config to disk.  Returns the config file path."""
    config_file = _config_path(config.show_name)
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=2))
    logger.info(f"saved show config: {config.show_name} -> {config_file}")
    return config_file


def list_shows() -> list[str]:
    """Return names of all shows with a config file."""
    shows_root = _shows_dir()
    if not os.path.isdir(shows_root):
        return []
    result = []
    for entry in sorted(os.listdir(shows_root)):
        config_file = os.path.join(shows_root, entry, "show_config.json")
        if os.path.isfile(config_file):
            result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Character operations
# ---------------------------------------------------------------------------

def add_character(
    show_name: str,
    character: CharacterDescription,
) -> ShowConfig:
    """Add or update a character in the show's bible and save."""
    config = load_show_config(show_name) or ShowConfig(show_name=show_name)

    # Replace existing or append
    existing_idx = None
    for i, c in enumerate(config.characters):
        if c.name.lower() == character.name.lower():
            existing_idx = i
            break

    if existing_idx is not None:
        config.characters[existing_idx] = character
        logger.info(f"updated character: {character.name} in {show_name}")
    else:
        config.characters.append(character)
        logger.info(f"added character: {character.name} to {show_name}")

    save_show_config(config)
    return config


def remove_character(show_name: str, character_name: str) -> ShowConfig | None:
    """Remove a character from the show's bible."""
    config = load_show_config(show_name)
    if not config:
        return None

    config.characters = [
        c for c in config.characters
        if c.name.lower() != character_name.lower()
    ]
    save_show_config(config)
    return config


# ---------------------------------------------------------------------------
# Location operations
# ---------------------------------------------------------------------------

def add_location(
    show_name: str,
    location: LocationDescription,
) -> ShowConfig:
    """Add or update a location in the show config and save."""
    config = load_show_config(show_name) or ShowConfig(show_name=show_name)

    existing_idx = None
    for i, loc in enumerate(config.locations):
        if loc.name.lower() == location.name.lower():
            existing_idx = i
            break

    if existing_idx is not None:
        config.locations[existing_idx] = location
    else:
        config.locations.append(location)

    save_show_config(config)
    return config


# ---------------------------------------------------------------------------
# Reference images
# ---------------------------------------------------------------------------

def get_character_reference(show_name: str, character_name: str) -> str | None:
    """Return path to a character's reference image, or None."""
    config = load_show_config(show_name)
    if not config:
        return None
    char = config.get_character(character_name)
    if not char or not char.reference_image:
        return None
    ref_path = char.reference_image
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(_characters_dir(show_name), ref_path)
    return ref_path if os.path.isfile(ref_path) else None


def get_location_reference(show_name: str, location_name: str) -> str | None:
    """Return path to a location's reference image, or None."""
    config = load_show_config(show_name)
    if not config:
        return None
    loc = config.get_location(location_name)
    if not loc or not loc.reference_image:
        return None
    ref_path = loc.reference_image
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(_locations_dir(show_name), ref_path)
    return ref_path if os.path.isfile(ref_path) else None


# ---------------------------------------------------------------------------
# Episode directory helpers
# ---------------------------------------------------------------------------

def episode_dir(show_name: str, episode_id: str) -> str:
    """Return (and create) the directory for a specific episode."""
    d = os.path.join(_episodes_dir(show_name), episode_id)
    os.makedirs(d, exist_ok=True)
    return d


def episode_shots_dir(show_name: str, episode_id: str) -> str:
    """Return (and create) the shots subdirectory for an episode."""
    d = os.path.join(episode_dir(show_name, episode_id), "shots")
    os.makedirs(d, exist_ok=True)
    return d


def episode_output_dir(show_name: str, episode_id: str) -> str:
    """Return (and create) the output subdirectory for an episode."""
    d = os.path.join(episode_dir(show_name, episode_id), "output")
    os.makedirs(d, exist_ok=True)
    return d
