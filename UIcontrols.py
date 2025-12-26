#!/usr/bin/env python3
"""UI Controls (Tkinter)

A small control panel for the simulation parameters + editable "protein" motifs.

Design notes
- Keep UI and simulation loosely coupled: Settings exposes properties; simulation reads them.
- Avoid global Tk root and store state inside the instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from tkinter import END, HORIZONTAL, INSERT, DoubleVar, Entry, Label, Listbox, Scale, Tk
from typing import Iterable, List

from Protein import Protein


@dataclass(frozen=True, slots=True)
class _ScaleSpec:
    label: str
    from_: float = 0.0
    to: float = 100.0
    length: int = 120
    x: int = 0
    y: int = 0


class Settings:
    """Tkinter settings panel."""

    def __init__(self, proteins: Iterable[Protein]) -> None:
        self.root = Tk(className="Control Panel")
        self.root.geometry("1200x320")

        self.current_selection: int = -1
        self._proteins = list(proteins)

        # --- Scales
        self._var_poison = DoubleVar()
        self._scale_poison = Scale(self.root, variable=self._var_poison, label="Poison Rate (%)", orient=HORIZONTAL, length=110, from_=0, to=100)
        self._scale_poison.place(y=5, x=5)

        self._var_food = DoubleVar()
        self._scale_food = Scale(self.root, variable=self._var_food, label="Food Rate (%)", orient=HORIZONTAL, length=110, from_=0, to=100)
        self._scale_food.place(y=5, x=125)

        self._var_min_bots = DoubleVar()
        self._scale_min_bots = Scale(self.root, variable=self._var_min_bots, label="Min Organisms", orient=HORIZONTAL, length=140, from_=0, to=200)
        self._scale_min_bots.place(y=5, x=245)

        self._var_mut = DoubleVar()
        self._scale_mut = Scale(self.root, variable=self._var_mut, label="Mutation Rate (x/1000)", orient=HORIZONTAL, length=140, from_=0, to=1000)
        self._scale_mut.place(y=5, x=395)

        self._var_steer = DoubleVar()
        self._scale_steer = Scale(self.root, variable=self._var_steer, label="Steering Weights (x/100)", orient=HORIZONTAL, length=140, from_=0, to=200)
        self._scale_steer.place(y=5, x=545)

        self._var_repr = DoubleVar()
        self._scale_repr = Scale(self.root, variable=self._var_repr, label="Reproduction Rate (x/10000)", orient=HORIZONTAL, length=160, from_=0, to=1000)
        self._scale_repr.place(y=5, x=695)

        self._var_max_vel = DoubleVar()
        self._scale_max_vel = Scale(self.root, variable=self._var_max_vel, label="Max Velocity", orient=HORIZONTAL, length=110, from_=0, to=20)
        self._scale_max_vel.place(y=5, x=865)

        self._var_dna_len = DoubleVar()
        self._scale_dna_len = Scale(self.root, variable=self._var_dna_len, label="DNA Length", orient=HORIZONTAL, length=110, from_=100, to=20000)
        self._scale_dna_len.place(y=5, x=985)

        self._var_health = DoubleVar()
        self._scale_health = Scale(self.root, variable=self._var_health, label="Initial Health", orient=HORIZONTAL, length=110, from_=1, to=300)
        self._scale_health.place(y=5, x=1105)

        self._var_max_poison = DoubleVar()
        self._scale_max_poison = Scale(self.root, variable=self._var_max_poison, label="Max Poison", orient=HORIZONTAL, length=110, from_=0, to=300)
        self._scale_max_poison.place(y=55, x=1105)

        # --- Protein editor
        Label(self.root, text="Protein Name").place(y=70, x=5)
        Label(self.root, text="Sequence").place(y=70, x=160)

        self._protein_name_entries: List[Entry] = []
        self._protein_seq_entries: List[Entry] = []
        for i, p in enumerate(self._proteins):
            en = Entry(self.root, width=18)
            en.insert(INSERT, p.name)
            en.place(y=95 + (i * 22), x=5)

            es = Entry(self.root, width=25)
            es.insert(INSERT, p.sequence)
            es.place(y=95 + (i * 22), x=160)

            self._protein_name_entries.append(en)
            self._protein_seq_entries.append(es)

        # --- Organism list panel
        self._listbox = Listbox(self.root)
        self._listbox.bind("<Double-1>", self._on_listbox_double_click)
        self._listbox.place(y=80, x=380, width=800, height=220)

    # -----------------
    # UI event handlers
    # -----------------
    def _on_listbox_double_click(self, _event) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        self.current_selection = int(sel[0])

    # -----------------
    # Protein controls
    # -----------------
    @property
    def user_proteins(self) -> List[Protein]:
        proteins: List[Protein] = []
        for name_entry, seq_entry in zip(self._protein_name_entries, self._protein_seq_entries, strict=False):
            name = name_entry.get().strip()
            seq = seq_entry.get().strip()
            if name and seq:
                proteins.append(Protein(name=name, sequence=seq))
        return proteins

    # -----------------
    # Listbox data
    # -----------------
    @property
    def listbox_widget(self) -> Listbox:
        return self._listbox

    @listbox_widget.setter
    def listbox_widget(self, entries: List[str]) -> None:
        self._listbox.delete(0, END)
        for e in entries:
            self._listbox.insert(END, e)

        # keep selection stable
        if 0 <= self.current_selection < self._listbox.size():
            self._listbox.selection_set(first=self.current_selection)
        else:
            self.current_selection = -1

    # -----------------
    # Simulation parameters
    # -----------------
    @property
    def poison_rate(self) -> float:
        return float(self._scale_poison.get()) / 100.0

    @poison_rate.setter
    def poison_rate(self, v: float) -> None:
        self._scale_poison.set(max(0.0, min(1.0, v)) * 100)

    @property
    def food_rate(self) -> float:
        return float(self._scale_food.get()) / 100.0

    @food_rate.setter
    def food_rate(self, v: float) -> None:
        self._scale_food.set(max(0.0, min(1.0, v)) * 100)

    @property
    def min_bot_count(self) -> int:
        return int(self._scale_min_bots.get())

    @min_bot_count.setter
    def min_bot_count(self, v: int) -> None:
        self._scale_min_bots.set(max(0, v))

    @property
    def mutation_rate(self) -> float:
        # UI stores 0..1000 => 0..1.0
        return float(self._scale_mut.get()) / 1000.0

    @mutation_rate.setter
    def mutation_rate(self, v: float) -> None:
        self._scale_mut.set(max(0.0, min(1.0, v)) * 1000)

    @property
    def steering_weights(self) -> float:
        # UI stores 0..200 => 0..2.0
        return float(self._scale_steer.get()) / 100.0

    @steering_weights.setter
    def steering_weights(self, v: float) -> None:
        self._scale_steer.set(max(0.0, v) * 100)

    @property
    def reproduction_rate(self) -> float:
        # UI stores 0..1000 => 0..0.1
        return float(self._scale_repr.get()) / 10000.0

    @reproduction_rate.setter
    def reproduction_rate(self, v: float) -> None:
        self._scale_repr.set(max(0.0, v) * 10000)

    @property
    def max_vel(self) -> float:
        return float(self._scale_max_vel.get())

    @max_vel.setter
    def max_vel(self, v: float) -> None:
        self._scale_max_vel.set(max(0.0, v))

    @property
    def dna_length(self) -> int:
        return int(self._scale_dna_len.get())

    @dna_length.setter
    def dna_length(self, v: int) -> None:
        self._scale_dna_len.set(max(100, v))

    @property
    def initial_health(self) -> float:
        return float(self._scale_health.get())

    @initial_health.setter
    def initial_health(self, v: float) -> None:
        self._scale_health.set(max(1.0, v))

    @property
    def max_poison(self) -> int:
        return int(self._scale_max_poison.get())

    @max_poison.setter
    def max_poison(self, v: int) -> None:
        self._scale_max_poison.set(max(0, v))
