"""Shared column names and OTTO constants."""

from __future__ import annotations

SESSION = "session"
AID = "aid"
TS = "ts"
TYPE = "type"
LABELS = "labels"
SESSION_TYPE = "session_type"

EVENT_TYPES = ("clicks", "carts", "orders")
EVENT_TYPE_WEIGHTS = {
    "clicks": 1.0,
    "carts": 6.0,
    "orders": 3.0,
}

