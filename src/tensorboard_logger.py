from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from tensorboardX import SummaryWriter  # type: ignore
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except ImportError:
        SummaryWriter = None


def _safe_float(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


class TensorboardRunLogger:
    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = enabled and SummaryWriter is not None
        self.status_message = "enabled"
        self._action_steps = {"P1": 0, "P2": 0}
        self._writer = None

        if not enabled:
            self.status_message = "disabled by configuration"
            return
        if SummaryWriter is None:
            self.status_message = (
                "unavailable: install tensorboardX or torch with tensorboard support"
            )
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(str(self.log_dir))

    def log_run_config(self, config: dict[str, Any]) -> None:
        if not self.enabled or self._writer is None:
            return
        try:
            self._writer.add_text(
                "run/config",
                json.dumps(config, indent=2, sort_keys=True),
                global_step=0,
            )
            self._writer.flush()
        except Exception as exc:
            self.enabled = False
            self.status_message = f"runtime_error: {exc}"

    def log_action_row(self, row: dict[str, str]) -> None:
        if not self.enabled or self._writer is None:
            return
        try:
            player = row.get("player_label", "unknown")
            step = self._action_steps.get(player, 0) + 1
            self._action_steps[player] = step
            prefix = f"action/{player}"

            scalar_fields = {
                "latency_ms": row.get("latency_ms"),
                "estimated_opponent_damage": row.get("estimated_opponent_damage"),
                "estimated_self_damage": row.get("estimated_self_damage"),
                "round_win_delta": row.get("round_win_delta"),
                "wins_p1_after": row.get("wins_p1_after"),
                "wins_p2_after": row.get("wins_p2_after"),
                "health_p1_after": row.get("health_p1_after"),
                "health_p2_after": row.get("health_p2_after"),
                "state_after_frame": row.get("state_after_frame"),
            }
            for name, value in scalar_fields.items():
                number = _safe_float(value)
                if number is not None:
                    self._writer.add_scalar(f"{prefix}/{name}", number, step)

            bool_fields = {
                "is_hallucination": row.get("is_hallucination"),
                "estimated_hit": row.get("estimated_hit"),
            }
            for name, value in bool_fields.items():
                parsed = _safe_bool(value)
                if parsed is not None:
                    self._writer.add_scalar(f"{prefix}/{name}", 1 if parsed else 0, step)

            details = row.get("decision_details", "")
            if details:
                self._writer.add_text(f"{prefix}/decision_details", details, step)
        except Exception as exc:
            self.enabled = False
            self.status_message = f"runtime_error: {exc}"

    def log_match_row(self, match_index: int, row: dict[str, str]) -> None:
        if not self.enabled or self._writer is None:
            return
        try:
            scalar_fields = {
                "duration_seconds": row.get("duration_seconds"),
                "wins_p1": row.get("wins_p1"),
                "wins_p2": row.get("wins_p2"),
                "health_p1": row.get("health_p1"),
                "health_p2": row.get("health_p2"),
                "p1_actions": row.get("p1_actions"),
                "p2_actions": row.get("p2_actions"),
                "p1_avg_latency_ms": row.get("p1_avg_latency_ms"),
                "p2_avg_latency_ms": row.get("p2_avg_latency_ms"),
                "p1_hallucinations": row.get("p1_hallucinations"),
                "p2_hallucinations": row.get("p2_hallucinations"),
                "p1_estimated_damage": row.get("p1_estimated_damage"),
                "p2_estimated_damage": row.get("p2_estimated_damage"),
                "p1_estimated_hits": row.get("p1_estimated_hits"),
                "p2_estimated_hits": row.get("p2_estimated_hits"),
                "p1_estimated_hit_rate": row.get("p1_estimated_hit_rate"),
                "p2_estimated_hit_rate": row.get("p2_estimated_hit_rate"),
            }
            for name, value in scalar_fields.items():
                number = _safe_float(value)
                if number is not None:
                    self._writer.add_scalar(f"match/{name}", number, match_index)

            result = row.get("result", "")
            result_map = {"P1": 1.0, "P2": 0.0, "draw_or_unknown": 0.5}
            if result in result_map:
                self._writer.add_scalar("match/p1_score", result_map[result], match_index)

            for model_key, elo_key in (
                ("model_p1", "p1_elo_after"),
                ("model_p2", "p2_elo_after"),
            ):
                model_name = row.get(model_key, "")
                elo_value = _safe_float(row.get(elo_key))
                if model_name and elo_value is not None:
                    self._writer.add_scalar(
                        f"elo/{_sanitize_tag(model_name)}",
                        elo_value,
                        match_index,
                    )

            self._writer.add_text(
                "match/summary_row",
                json.dumps(row, indent=2, sort_keys=True),
                match_index,
            )
            self._writer.flush()
        except Exception as exc:
            self.enabled = False
            self.status_message = f"runtime_error: {exc}"

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass
