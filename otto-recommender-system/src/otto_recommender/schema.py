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

# Stable feature contract shared by retrieval, ranking, and inference jobs.
ITEM_FEATURES = (
    "total_interactions", "click_count", "cart_count", "order_count",
    "recent_24h_interactions", "conversion_rate", "item_cart_conversion_rate",
    "item_buy_conversion_rate", "item_cart_to_order_rate", "item_funnel_dropoff_rate",
)
SESSION_FEATURES = (
    "session_length", "session_unique_items", "session_first_ts", "session_last_ts",
    "session_duration", "session_cart_count", "session_order_count",
)
CROSS_FEATURES = (
    "local_interaction_count", "local_click_count", "local_cart_count", "local_order_count",
    "is_repeated_item", "delta_t_filled", "graph_weight_total",
)
FEATURE_DEFAULTS = {"delta_t_filled": 9_999_999}
