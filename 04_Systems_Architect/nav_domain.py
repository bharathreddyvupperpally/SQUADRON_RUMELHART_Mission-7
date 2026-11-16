"""nav_domain.py - Shared toy "Sagittarius A* navigation" reasoning domain.

A trajectory is a chain of arithmetic thoughts that steers a probe from a
start value to a target value.  Every thought is written as an equation
("12 * 3 = 36"), so a Process Reward Model can verify it step by step.

Event Horizon: any value below 0 or above 3 * target is a point of no
return (a "dead" state).  The AURA proposer can also *hallucinate*: with
probability `hallucination_rate` a thought claims a wrong result, and the
state then follows the wrong claim, so the error compounds.

Pure standard library.  This file is identical in folders 01 and 02.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

OPS: Tuple[Tuple[str, int], ...] = (("+", 2), ("+", 5), ("*", 2), ("*", 3), ("-", 3))


def apply_op(value: int, op: str, k: int) -> int:
    if op == "+":
        return value + k
    if op == "-":
        return value - k
    if op == "*":
        return value * k
    raise ValueError(f"unknown op {op!r}")


@dataclass(frozen=True)
class Thought:
    text: str                 # e.g. "12 * 3 = 36"
    state: Tuple[int, int]    # (value, depth) AFTER the thought
    hallucinated: bool = False  # ground truth flag, used only for logging


class NavigationProblem:
    """Reach `target` from `start` within `max_depth` thoughts."""

    def __init__(self, start: int = 5, target: int = 47, max_depth: int = 6,
                 hallucination_rate: float = 0.0, seed: int = 7):
        self.start = start
        self.target = target
        self.max_depth = max_depth
        self.hallucination_rate = hallucination_rate
        self.seed = seed
        self.horizon = 3 * target

    # ---- problem interface used by ToT and MCTS ------------------------
    def initial_state(self) -> Tuple[int, int]:
        return (self.start, 0)

    def is_goal(self, state) -> bool:
        return state[0] == self.target

    def is_dead(self, state) -> bool:
        """Beyond the event horizon: unrecoverable."""
        return state[0] < 0 or state[0] > self.horizon

    def is_terminal(self, state) -> bool:
        return self.is_goal(state) or self.is_dead(state) or state[1] >= self.max_depth

    def value(self, state) -> float:
        """Heuristic value in [0, 1]; 1 only at the goal, 0 when dead."""
        if self.is_goal(state):
            return 1.0
        if self.is_dead(state):
            return 0.0
        return max(0.01, 1.0 - abs(self.target - state[0]) / self.target) * 0.95

    def propose(self, state) -> List[Thought]:
        """The 'LLM policy': one candidate thought per operator."""
        v, d = state
        out = []
        for op, k in OPS:
            true = apply_op(v, op, k)
            claimed, hallu = true, False
            h = int(hashlib.md5(f"{self.seed}|{v}|{d}|{op}{k}".encode()).hexdigest(), 16)
            if (h % 1000) / 1000.0 < self.hallucination_rate:
                offset = (h // 1000) % 6 + 1            # 1..6
                claimed = true + (offset if (h // 7) % 2 else -offset)
                hallu = claimed != true
            out.append(Thought(f"{v} {op} {k} = {claimed}", (claimed, d + 1), hallu))
        return out

    # ---- ground truth helper (tests / sanity only) ---------------------
    def shortest_solution(self) -> Optional[int]:
        """BFS over the TRUE (non-hallucinated) dynamics; returns depth or None."""
        seen = {self.start}
        q = deque([(self.start, 0)])
        while q:
            v, d = q.popleft()
            if v == self.target:
                return d
            if d >= self.max_depth:
                continue
            for op, k in OPS:
                n = apply_op(v, op, k)
                if 0 <= n <= self.horizon and n not in seen:
                    seen.add(n)
                    q.append((n, d + 1))
        return None
