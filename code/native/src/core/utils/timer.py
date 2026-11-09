# core/utils/timer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


def format_seconds(sec: Optional[float]) -> str:
    try:
        x = max(0, int(round(float(sec or 0.0))))
    except Exception:
        x = 0
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


@dataclass
class Timer:
    _t0: float = field(default_factory=time.perf_counter)
    _acc: Dict[str, float] = field(default_factory=dict)

    def tic(self) -> float:
        return time.perf_counter()

    def toc(self, key: str, t_start: float) -> None:
        dt = time.perf_counter() - t_start
        self._acc[key] = self._acc.get(key, 0.0) + dt

    def items(self):
        return sorted(self._acc.items(), key=lambda kv: kv[0])

    def as_dict(self):
        return dict(self.items())

    def pretty(self) -> str:
        parts = [f"{k}={v:.4f}s" for k, v in self.items()]
        return " | ".join(parts)


@dataclass
class ProgressTracker:
    total: int
    name: str = "Progress"
    print_every: int = 1
    start_time: float = field(default_factory=time.time)
    done: int = 0

    def step(self, n: int = 1) -> None:
        self.done += int(n)

    def elapsed(self) -> float:
        return max(0.0, time.time() - self.start_time)

    def avg_per_task(self) -> float:
        return self.elapsed() / max(1, int(self.done))

    def eta(self) -> float:
        return self.avg_per_task() * max(0, int(self.total) - int(self.done))

    def should_print(self) -> bool:
        return (self.done % max(1, int(self.print_every)) == 0) or (self.done >= int(self.total))

    def summary(self, extra: str = "") -> str:
        pct = 100.0 * float(self.done) / float(max(1, int(self.total)))
        msg = (
            f"[{self.name}] {self.done}/{self.total} ({pct:.1f}%)"
            f" | elapsed={format_seconds(self.elapsed())}"
            f" | avg/task={self.avg_per_task():.2f}s"
            f" | ETA={format_seconds(self.eta())}"
        )
        if extra:
            msg += f" | {extra}"
        return msg

    def print(self, extra: str = "", force: bool = False) -> None:
        if force or self.should_print():
            print(self.summary(extra=extra), flush=True)
