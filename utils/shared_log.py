"""Utilities for maintaining a shared append-only agent log."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class LogEntry:
    """A single entry in the shared log."""

    agent: str
    type: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    needs: List[str] = field(default_factory=list)
    resolves: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the entry."""

        return {
            "agent": self.agent,
            "type": self.type,
            "content": self.content,
            "metadata": dict(self.metadata),
            "needs": list(self.needs),
            "resolves": list(self.resolves),
        }


class SharedLog:
    """An append-only store that tracks pending coordination needs."""

    def __init__(self, entry_transform=None) -> None:
        """
        Parameters
        ----------
        entry_transform:
            Optional callable ``f(LogEntry) -> LogEntry`` applied before each entry is
            stored. Useful for instrumentation (e.g., fault injection) without touching
            agent code paths.
        """
        self._entries: List[LogEntry] = []
        self._pending: Counter[str] = Counter()
        self._transform = entry_transform

    def append(
        self,
        agent: str,
        entry_type: str,
        content: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        needs: Optional[Iterable[str]] = None,
        resolves: Optional[Iterable[str]] = None,
    ) -> LogEntry:
        """Append a new entry to the log."""

        entry = LogEntry(
            agent=agent,
            type=entry_type,
            content=content,
            metadata=dict(metadata or {}),
            needs=list(needs or []),
            resolves=list(resolves or []),
        )
        if self._transform:
            entry = self._transform(entry)
        self._entries.append(entry)

        for need in entry.needs:
            if need:
                self._pending[need] += 1

        for resolved in entry.resolves:
            if resolved and self._pending.get(resolved):
                self._pending[resolved] -= 1
                if self._pending[resolved] <= 0:
                    del self._pending[resolved]

        return entry

    def pending_needs(self) -> List[str]:
        """Return a sorted list of outstanding coordination needs."""

        return sorted(self._pending.elements())

    def has_pending(self, need: str) -> bool:
        """Return ``True`` if ``need`` is still outstanding."""

        return self._pending.get(need, 0) > 0

    def latest(self, entry_type: str) -> Optional[LogEntry]:
        """Return the most recent entry of ``entry_type`` if available."""

        for entry in reversed(self._entries):
            if entry.type == entry_type:
                return entry
        return None

    def entries(self) -> List[LogEntry]:
        """Return all log entries."""

        return list(self._entries)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return the log entries as dictionaries."""

        return [entry.to_dict() for entry in self._entries]

    def __iter__(self) -> Iterator[LogEntry]:  # pragma: no cover - simple delegation
        return iter(self._entries)
