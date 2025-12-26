#!/usr/bin/env python3
"""Protein primitives.

This project uses short amino-acid motifs as "proteins" that map to organism traits
(perception radius and steering weights).

The simulation detects these motifs inside an organism's translated amino-acid
sequence and derives the organism's behavior parameters from the counts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Protein:
    """A named amino-acid motif."""

    name: str
    sequence: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Protein.name must be non-empty")
        if not self.sequence:
            raise ValueError("Protein.sequence must be non-empty")
