#!/usr/bin/env python3
"""Evolutionary mutation simulation (pygame + Tkinter)

This project simulates a population of organisms with simple "DNA" strings.
Each DNA is translated into an amino-acid sequence (toy translation), which is
scanned for motif "proteins". Those proteins influence organism behavior:
- food vs poison perception radii
- steering tendencies toward/away from items

The simulation is intentionally simplified for visualization/learning.

Run:
  python Dna_Simulation.py

Dependencies:
  pip install pygame numpy
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from pygame import gfxdraw

from Protein import Protein
from UIcontrols import Settings

# ----------------------------
# Genetics / translation config
# ----------------------------

START_CODON = "ATG"
STOP_CODONS = {"TAA", "TAG", "TGA"}

CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "CTT": "L", "ATT": "I", "GTT": "V",
    "TTC": "F", "CTC": "L", "ATC": "I", "GTC": "V",
    "TTA": "L", "CTA": "L", "ATA": "I", "GTA": "V",
    "TTG": "L", "CTG": "L", "ATG": "M", "GTG": "V",
    "TCT": "S", "CCT": "P", "ACT": "T", "GCT": "A",
    "TCC": "S", "CCC": "P", "ACC": "T", "GCC": "A",
    "TCA": "S", "CCA": "P", "ACA": "T", "GCA": "A",
    "TCG": "S", "CCG": "P", "ACG": "T", "GCG": "A",
    "TAT": "Y", "CAT": "H", "AAT": "N", "GAT": "D",
    "TAC": "Y", "CAC": "H", "AAC": "N", "GAC": "D",
    "CAA": "Q", "AAA": "K", "GAA": "E",
    "CAG": "Q", "AAG": "K", "GAG": "E",
    "TGT": "C", "CGT": "R", "AGT": "S", "GGT": "G",
    "TGC": "C", "CGC": "R", "AGC": "S", "GGC": "G",
    "CGA": "R", "AGA": "R", "GGA": "G",
    "TGG": "W", "CGG": "R", "AGG": "R", "GGG": "G",
}

# ----------------------------
# Simulation configuration
# ----------------------------

@dataclass(slots=True)
class SimulationConfig:
    width: int = 800
    height: int = 400
    fps: int = 60

    boundary_size: int = 10
    initial_health: float = 100.0
    max_vel: float = 10.0

    # population / environment
    min_bot_count: int = 10
    starting_bot_count: int = 20
    food_rate: float = 0.20
    poison_rate: float = 0.05
    max_poison: int = 25

    # genetics
    dna_length: int = 1000
    mutation_rate: float = 0.005
    reproduction_rate: float = 0.0008
    steering_weights: float = 0.20

    # organism behavior
    initial_perception_radius: float = 1.0
    initial_max_force: float = 0.02
    nutrition_food: float = 20.0
    nutrition_poison: float = -80.0


DEFAULT_PROTEINS: List[Protein] = [
    Protein("food_perception", "ST"),
    Protein("poison_perception", "AI"),
    Protein("red_tail+", "QR"),
    Protein("red_tail-", "RR"),
    Protein("green_tail+", "PR"),
    Protein("green_tail-", "LR"),
]

# ----------------------------
# Vector helpers
# ----------------------------

def magnitude(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------
# DNA model
# ----------------------------

class DNA:
    """A mutable DNA string + derived protein sequence."""

    _alphabet = ("A", "C", "G", "T")

    def __init__(self, cfg: SimulationConfig, protein_library: Sequence[Protein], base: Optional["DNA"] = None) -> None:
        self.cfg = cfg
        self.genetic_code: str = base.genetic_code if base else self._random_code(cfg.dna_length)
        if base:
            self._mutate()
        self.protein_sequence: str = self._translate()
        self.proteins: List[Protein] = self._scan_proteins(protein_library)

    def replicate(self, protein_library: Sequence[Protein]) -> "DNA":
        return DNA(self.cfg, protein_library=protein_library, base=self)

    @staticmethod
    def _random_code(length: int) -> str:
        return "".join(random.choice(DNA._alphabet) for _ in range(length))

    def _mutate(self) -> None:
        # expected number of operations proportional to dna_length * mutation_rate
        ops = max(1, int(self.cfg.dna_length * self.cfg.mutation_rate))
        code = list(self.genetic_code)

        for _ in range(ops):
            r = random.random()
            if r < 0.40:
                # alter nucleotide
                i = random.randrange(len(code))
                old = code[i]
                choices = [c for c in self._alphabet if c != old]
                code[i] = random.choice(choices)
            elif r < 0.80:
                # insert nucleotide
                i = random.randrange(len(code) + 1)
                code.insert(i, random.choice(self._alphabet))
            elif r < 0.90:
                # remove nucleotide (if possible)
                if len(code) > 1:
                    i = random.randrange(len(code))
                    code.pop(i)
            else:
                # rare duplication
                if random.random() < 0.002:
                    code = code + code

        self.genetic_code = "".join(code)

    def _translate(self) -> str:
        """Toy translation: find in-frame ORFs starting at ATG, end at first in-frame stop."""
        seq = self.genetic_code
        aa_out: List[str] = []
        n = len(seq)

        i = 0
        while True:
            start = seq.find(START_CODON, i)
            if start < 0:
                break

            # translate codons in-frame after start
            j = start
            aa_out.append("&")  # ORF marker start
            j += 3
            while j + 2 < n:
                codon = seq[j : j + 3]
                if codon in STOP_CODONS:
                    aa_out.append("_")  # ORF marker end
                    j += 3
                    break
                aa_out.append(CODON_TABLE.get(codon, "?"))
                j += 3

            i = j if j > start else start + 3

        return "".join(aa_out)

    def _scan_proteins(self, protein_library: Sequence[Protein]) -> List[Protein]:
        found: List[Protein] = []
        ps = self.protein_sequence
        if not ps:
            return found
        for p in protein_library:
            count = ps.count(p.sequence)
            if count > 0:
                found.extend([p] * count)
        return found

    def protein_count(self, name: str) -> int:
        return sum(1 for p in self.proteins if p.name == name)


# ----------------------------
# Organism model
# ----------------------------

class Organism:
    def __init__(
        self,
        cfg: SimulationConfig,
        protein_library: Sequence[Protein],
        x: float,
        y: float,
        base_dna: Optional[DNA] = None,
    ) -> None:
        self.cfg = cfg
        self.position = np.array([x, y], dtype=np.float64)
        self.velocity = np.array(
            [random.uniform(-cfg.max_vel, cfg.max_vel), random.uniform(-cfg.max_vel, cfg.max_vel)],
            dtype=np.float64,
        )
        self.acceleration = np.zeros(2, dtype=np.float64)

        self.health: float = cfg.initial_health
        self.age: int = 1

        self.max_vel: float = 2.0
        self.max_force: float = 0.5

        self.dna: DNA = (base_dna.replicate(protein_library) if base_dna else DNA(cfg, protein_library))
        self._derive_traits()

    # ----- traits
    def _derive_traits(self) -> None:
        # perception scales with protein motif counts + tiny randomness
        self.food_perception = 10.0 * self.dna.protein_count("food_perception") + random.uniform(0.0, self.cfg.initial_perception_radius)
        self.poison_perception = 10.0 * self.dna.protein_count("poison_perception") + random.uniform(0.0, self.cfg.initial_perception_radius)

        # steering weights influenced by proteins; can be negative
        gw = self.cfg.steering_weights
        self.green_force = 0.0
        self.red_force = 0.0

        self.green_force += gw * self.dna.protein_count("green_tail+") + random.uniform(0.0, self.cfg.initial_max_force)
        self.green_force += -gw * self.dna.protein_count("green_tail-") + random.uniform(0.0, self.cfg.initial_max_force)

        self.red_force += gw * self.dna.protein_count("red_tail+") + random.uniform(0.0, self.cfg.initial_max_force)
        self.red_force += -gw * self.dna.protein_count("red_tail-") + random.uniform(0.0, self.cfg.initial_max_force)

    # ----- physics
    def apply_force(self, f: np.ndarray) -> None:
        self.acceleration += f

    def seek(self, target_xy: Sequence[float]) -> np.ndarray:
        target = np.array(target_xy, dtype=np.float64)
        desired = normalize(target - self.position) * self.max_vel
        steer = normalize(desired - self.velocity) * self.max_force
        return steer

    def boundaries(self) -> None:
        b = self.cfg.boundary_size
        w = self.cfg.width
        h = self.cfg.height

        x, y = self.position
        desired: Optional[np.ndarray] = None

        if x < b:
            desired = np.array([self.max_vel, self.velocity[1]])
        elif x > w - b:
            desired = np.array([-self.max_vel, self.velocity[1]])

        if y < b:
            desired = np.array([self.velocity[0], self.max_vel]) if desired is None else np.array([desired[0], self.max_vel])
        elif y > h - b:
            desired = np.array([self.velocity[0], -self.max_vel]) if desired is None else np.array([desired[0], -self.max_vel])

        if desired is not None:
            steer = normalize(desired - self.velocity) * self.max_force
            self.apply_force(steer)

    def update(self, initial_health: float) -> None:
        self.velocity += self.acceleration
        self.velocity = normalize(self.velocity) * self.max_vel
        self.position += self.velocity
        self.acceleration *= 0.0

        # health decays each tick
        self.health -= 0.2
        self.health = min(initial_health, self.health)
        self.age += 1

    # ----- behaviors
    def reproduce(self, population: List["Organism"], protein_library: Sequence[Protein]) -> None:
        if random.random() < self.cfg.reproduction_rate:
            population.append(Organism(self.cfg, protein_library, float(self.position[0]), float(self.position[1]), base_dna=self.dna))

    def is_dead(self) -> bool:
        return self.health <= 0.0

    def drop_food_if_inside(self, food: List[np.ndarray]) -> None:
        b = self.cfg.boundary_size
        if b < self.position[0] < self.cfg.width - b and b < self.position[1] < self.cfg.height - b:
            food.append(self.position.copy())

    def _eat_from(self, items: List[np.ndarray], nutrition: float, perception: float, force_scale: float) -> None:
        if not items:
            return

        closest_idx: Optional[int] = None
        closest_dist = float("inf")

        # iterate backwards so popping is safe
        for idx in range(len(items) - 1, -1, -1):
            item = items[idx]
            dist = math.hypot(self.position[0] - item[0], self.position[1] - item[1])

            if dist < 5.0:
                items.pop(idx)
                self.health += nutrition
                continue

            if dist < closest_dist:
                closest_dist = dist
                closest_idx = idx

        if closest_idx is None:
            return

        if closest_dist < perception:
            steer = self.seek(items[closest_idx])
            steer *= force_scale
            steer = normalize(steer) * self.max_force
            self.apply_force(steer)

    def eat(self, food: List[np.ndarray], poison: List[np.ndarray]) -> None:
        self._eat_from(food, self.cfg.nutrition_food, self.food_perception, self.green_force)
        self._eat_from(poison, self.cfg.nutrition_poison, self.poison_perception, self.red_force)

    # ----- rendering
    def _health_color(self, initial_health: float) -> Tuple[int, int, int]:
        # red -> green as health increases
        p = clamp01(self.health / max(1e-6, initial_health))
        r = int(clamp01(1.0 - p) * 255)
        g = int(clamp01(p) * 255)
        return (r, g, 0)

    def draw(self, surface: pygame.Surface, index: int, selected_index: int, colors: Dict[str, Tuple[int, int, int]], initial_health: float) -> None:
        x, y = int(self.position[0]), int(self.position[1])
        col = self._health_color(initial_health)
        fill = colors["blue"] if selected_index == index else col

        gfxdraw.aacircle(surface, x, y, 10, col)
        gfxdraw.filled_circle(surface, x, y, 10, fill)

        # perception radii
        pygame.draw.circle(surface, colors["green"], (x, y), abs(int(self.food_perception)), max(1, abs(int(min(2, self.food_perception)))))
        pygame.draw.circle(surface, colors["red"], (x, y), abs(int(self.poison_perception)), max(1, abs(int(min(2, self.poison_perception)))))

        # steering direction hints
        pygame.draw.line(surface, colors["green"], (x, y), (int(x + self.velocity[0] * self.green_force * 25), int(y + self.velocity[1] * self.green_force * 25)), 3)
        pygame.draw.line(surface, colors["red"], (x, y), (int(x + self.velocity[0] * self.red_force * 25), int(y + self.velocity[1] * self.red_force * 25)), 2)


# ----------------------------
# Simulation runtime
# ----------------------------

class Simulation:
    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg

        self.colors = {
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (65, 131, 196),
        }

        self.organisms: List[Organism] = []
        self.food: List[np.ndarray] = []
        self.poison: List[np.ndarray] = []

        self.oldest_ever: int = 0
        self.protein_library: List[Protein] = list(DEFAULT_PROTEINS)

        # UI
        self.settings = Settings(self.protein_library)
        self._apply_defaults_to_ui()

    def _apply_defaults_to_ui(self) -> None:
        s = self.settings
        s.poison_rate = self.cfg.poison_rate
        s.food_rate = self.cfg.food_rate
        s.min_bot_count = self.cfg.min_bot_count
        s.max_vel = self.cfg.max_vel
        s.initial_health = self.cfg.initial_health
        s.max_poison = self.cfg.max_poison
        s.mutation_rate = self.cfg.mutation_rate
        s.reproduction_rate = self.cfg.reproduction_rate
        s.steering_weights = self.cfg.steering_weights
        s.dna_length = self.cfg.dna_length

    def _sync_cfg_from_ui(self) -> None:
        # allows live tuning
        s = self.settings
        self.cfg.poison_rate = float(s.poison_rate)
        self.cfg.food_rate = float(s.food_rate)
        self.cfg.min_bot_count = int(s.min_bot_count)
        self.cfg.max_vel = float(s.max_vel)
        self.cfg.initial_health = float(s.initial_health)
        self.cfg.max_poison = int(s.max_poison)
        self.cfg.mutation_rate = float(s.mutation_rate)
        self.cfg.reproduction_rate = float(s.reproduction_rate)
        self.cfg.steering_weights = float(s.steering_weights)
        self.cfg.dna_length = int(s.dna_length)

    def _seed_population(self) -> None:
        for _ in range(self.cfg.starting_bot_count):
            self.organisms.append(
                Organism(
                    self.cfg,
                    self.protein_library,
                    random.uniform(0, self.cfg.width),
                    random.uniform(0, self.cfg.height),
                )
            )

    def _spawn_items(self) -> None:
        b = self.cfg.boundary_size
        if random.random() < self.cfg.food_rate:
            self.food.append(np.array([random.uniform(b, self.cfg.width - b), random.uniform(b, self.cfg.height - b)], dtype=np.float64))

        if random.random() < self.cfg.poison_rate:
            self.poison.append(np.array([random.uniform(b, self.cfg.width - b), random.uniform(b, self.cfg.height - b)], dtype=np.float64))

        while len(self.poison) > self.cfg.max_poison:
            self.poison.pop(0)

        # maintain a minimum population, plus a small chance of spontaneous spawn
        if len(self.organisms) < self.cfg.min_bot_count or random.random() < 0.0001:
            self.organisms.append(Organism(self.cfg, self.protein_library, random.uniform(0, self.cfg.width), random.uniform(0, self.cfg.height)))

    def _update_listbox(self) -> None:
        entries: List[str] = []
        for idx, org in enumerate(self.organisms, start=1):
            parts = [p.sequence for p in org.dna.proteins] or ["None"]
            entries.append(f"{idx}) Proteins: {'-'.join(parts)}    Seq: {org.dna.protein_sequence}")
        self.settings.listbox_widget = entries

    def run(self) -> None:
        pygame.init()
        display = pygame.display.set_mode((self.cfg.width, self.cfg.height))
        clock = pygame.time.Clock()

        self._seed_population()

        # periodic events
        EVENT_TITLE = pygame.USEREVENT + 1
        EVENT_PROTEINS = pygame.USEREVENT + 2
        EVENT_LISTBOX = pygame.USEREVENT + 3

        pygame.time.set_timer(EVENT_TITLE, 500)
        pygame.time.set_timer(EVENT_PROTEINS, 5000)
        pygame.time.set_timer(EVENT_LISTBOX, 2000)

        running = True
        while running:
            # UI step
            self.settings.root.update()
            self._sync_cfg_from_ui()

            display.fill(self.colors["black"])
            self._spawn_items()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == EVENT_TITLE:
                    self.settings.root.title(f"Number of Organisms: {len(self.organisms)}")
                elif event.type == EVENT_PROTEINS:
                    # re-load proteins from UI
                    self.protein_library = self.settings.user_proteins or list(DEFAULT_PROTEINS)
                elif event.type == EVENT_LISTBOX:
                    self._update_listbox()

            # simulation step
            selected = self.settings.current_selection
            for idx in range(len(self.organisms) - 1, -1, -1):
                org = self.organisms[idx]

                org.eat(self.food, self.poison)
                org.boundaries()
                org.update(initial_health=self.cfg.initial_health)

                if org.age > self.oldest_ever:
                    self.oldest_ever = org.age

                org.draw(display, index=idx, selected_index=selected, colors=self.colors, initial_health=self.cfg.initial_health)

                if org.is_dead():
                    org.drop_food_if_inside(self.food)
                    self.organisms.pop(idx)
                else:
                    org.reproduce(self.organisms, self.protein_library)

            # render environment items
            for f in self.food:
                pygame.draw.circle(display, self.colors["green"], (int(f[0]), int(f[1])), 3)
            for p in self.poison:
                pygame.draw.circle(display, self.colors["red"], (int(p[0]), int(p[1])), 3)

            pygame.display.update()
            clock.tick(self.cfg.fps)

        pygame.quit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evolutionary DNA mutation simulation (pygame + Tkinter).")
    p.add_argument("--seed", type=int, default=None, help="Random seed (for reproducible runs)")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--fps", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    cfg = SimulationConfig(width=args.width, height=args.height, fps=args.fps)
    Simulation(cfg).run()


if __name__ == "__main__":
    main()
