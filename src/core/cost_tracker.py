"""
Cost Tracking System

Tracks token usage and estimated costs across sessions.

Features:
- Real-time token tracking
- Cost estimation based on model pricing
- Daily/session/project statistics
- Budget limits and warnings
- Usage history and trends
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ========================================
# Model Pricing (USD per 1M tokens)
# Updated: January 2026 - Latest models only
# ========================================

MODEL_PRICING = {
    # ========== Google Gemini 3 Series (Latest) ==========
    "gemini-3-pro": {"input": 2.00, "output": 12.00},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash": {"input": 0.50, "output": 3.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},

    # ========== OpenAI GPT-5 Series (Latest) ==========
    "gpt-5": {"input": 5.00, "output": 15.00},
    "gpt-5-pro": {"input": 10.00, "output": 30.00},
    "gpt-5.2": {"input": 5.00, "output": 15.00},
    "gpt-5-mini": {"input": 2.50, "output": 2.00},

    # ========== Anthropic Claude 4.5 Series (Latest) ==========
    "claude-opus-4.5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},

    # Default fallback
    "default": {"input": 0.50, "output": 2.00},
}


@dataclass
class UsageRecord:
    """A single usage record."""

    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    session_id: str
    project: str
    operation: str = ""  # e.g., "chat", "summarize", "subagent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "session_id": self.session_id,
            "project": self.project,
            "operation": self.operation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageRecord":
        return cls(
            timestamp=data["timestamp"],
            model=data["model"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cost_usd=data["cost_usd"],
            session_id=data["session_id"],
            project=data["project"],
            operation=data.get("operation", ""),
        )


@dataclass
class UsageStats:
    """Aggregated usage statistics."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0
    period_start: float = 0
    period_end: float = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_tokens_per_request(self) -> float:
        if self.request_count == 0:
            return 0
        return self.total_tokens / self.request_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "request_count": self.request_count,
            "avg_tokens_per_request": round(self.avg_tokens_per_request, 1),
            "period_start": datetime.fromtimestamp(self.period_start).isoformat()
            if self.period_start
            else None,
            "period_end": datetime.fromtimestamp(self.period_end).isoformat()
            if self.period_end
            else None,
        }


@dataclass
class BudgetConfig:
    """Budget configuration."""

    daily_limit_usd: float | None = None  # Daily spending limit
    session_limit_usd: float | None = None  # Per-session limit
    warning_threshold: float = 0.8  # Warn at 80% of limit
    enabled: bool = True


class CostTracker:
    """
    Tracks token usage and costs.

    Usage:
        tracker = CostTracker(project="myproject")

        # Track usage
        tracker.track(
            model="gemini-2.0-flash",
            input_tokens=1000,
            output_tokens=500,
            session_id="abc123"
        )

        # Get statistics
        stats = tracker.get_session_stats("abc123")
        daily = tracker.get_daily_stats()

        # Check budget
        if tracker.check_budget():
            print("Within budget")

        # Get cost summary
        summary = tracker.get_cost_summary()
    """

    def __init__(
        self,
        project: str = "default",
        session_id: str = "",
        storage_dir: str = ".doraemon/usage",
        budget: BudgetConfig | None = None,
    ):
        """
        Initialize cost tracker.

        Args:
            project: Project name
            session_id: Current session ID
            storage_dir: Directory for usage data
            budget: Budget configuration
        """
        self.project = project
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget or BudgetConfig()

        # Current session tracking
        self._session_records: list[UsageRecord] = []
        self._session_start = time.time()

        # Load today's records
        self._today_records: list[UsageRecord] = []
        self._load_today_records()

    def _get_today_file(self) -> Path:
        """Get path for today's usage file."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.storage_dir / f"{today}.json"

    def _load_today_records(self):
        """Load today's usage records."""
        path = self._get_today_file()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._today_records = [
                    UsageRecord.from_dict(r) for r in data.get("records", [])
                ]
            except Exception as e:
                logger.error(f"Failed to load usage records: {e}")

    def _save_today_records(self):
        """Save today's usage records."""
        path = self._get_today_file()
        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "records": [r.to_dict() for r in self._today_records],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Calculate cost for token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        # Find pricing for model
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            # Try partial match
            for model_name, price in MODEL_PRICING.items():
                if model_name in model or model in model_name:
                    pricing = price
                    break

        if not pricing:
            pricing = MODEL_PRICING["default"]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        session_id: str | None = None,
        operation: str = "chat",
    ) -> UsageRecord:
        """
        Track a usage event.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            session_id: Session ID (uses current if not provided)
            operation: Operation type

        Returns:
            Created UsageRecord
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            session_id=session_id or self.session_id,
            project=self.project,
            operation=operation,
        )

        # Add to session and daily records
        self._session_records.append(record)
        self._today_records.append(record)

        # Save to disk
        self._save_today_records()

        logger.debug(
            f"Tracked usage: {input_tokens}+{output_tokens} tokens, ${cost:.4f}"
        )

        return record

    def get_session_stats(self, session_id: str | None = None) -> UsageStats:
        """Get statistics for a session."""
        sid = session_id or self.session_id
        records = [r for r in self._session_records if r.session_id == sid]

        if not records:
            return UsageStats()

        return UsageStats(
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
            total_cost_usd=sum(r.cost_usd for r in records),
            request_count=len(records),
            period_start=min(r.timestamp for r in records),
            period_end=max(r.timestamp for r in records),
        )

    def get_daily_stats(self, date: str | None = None) -> UsageStats:
        """
        Get statistics for a day.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today
        """
        if date:
            # Load specific day
            path = self.storage_dir / f"{date}.json"
            if not path.exists():
                return UsageStats()
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                records = [UsageRecord.from_dict(r) for r in data.get("records", [])]
            except Exception:
                return UsageStats()
        else:
            records = self._today_records

        if not records:
            return UsageStats()

        return UsageStats(
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
            total_cost_usd=sum(r.cost_usd for r in records),
            request_count=len(records),
            period_start=min(r.timestamp for r in records),
            period_end=max(r.timestamp for r in records),
        )

    def get_project_stats(self, days: int = 30) -> UsageStats:
        """Get statistics for project over specified days."""
        all_records = []

        # Load records from each day
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            path = self.storage_dir / f"{date}.json"

            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    records = [
                        UsageRecord.from_dict(r)
                        for r in data.get("records", [])
                        if UsageRecord.from_dict(r).project == self.project
                    ]
                    all_records.extend(records)
                except Exception:
                    continue

        if not all_records:
            return UsageStats()

        return UsageStats(
            total_input_tokens=sum(r.input_tokens for r in all_records),
            total_output_tokens=sum(r.output_tokens for r in all_records),
            total_cost_usd=sum(r.cost_usd for r in all_records),
            request_count=len(all_records),
            period_start=min(r.timestamp for r in all_records),
            period_end=max(r.timestamp for r in all_records),
        )

    def check_budget(self) -> dict[str, Any]:
        """
        Check budget status.

        Returns:
            Dict with budget status:
            - within_budget: bool
            - daily_usage: float
            - daily_limit: float | None
            - session_usage: float
            - session_limit: float | None
            - warning: str | None
        """
        if not self.budget.enabled:
            return {"within_budget": True, "warning": None}

        daily_stats = self.get_daily_stats()
        session_stats = self.get_session_stats()

        result = {
            "within_budget": True,
            "daily_usage": daily_stats.total_cost_usd,
            "daily_limit": self.budget.daily_limit_usd,
            "session_usage": session_stats.total_cost_usd,
            "session_limit": self.budget.session_limit_usd,
            "warning": None,
        }

        # Check daily limit
        if self.budget.daily_limit_usd and self.budget.daily_limit_usd > 0:
            daily_pct = daily_stats.total_cost_usd / self.budget.daily_limit_usd

            if daily_pct >= 1.0:
                result["within_budget"] = False
                result["warning"] = (
                    f"Daily budget exceeded: ${daily_stats.total_cost_usd:.2f} / "
                    f"${self.budget.daily_limit_usd:.2f}"
                )
            elif daily_pct >= self.budget.warning_threshold:
                result["warning"] = (
                    f"Approaching daily budget: ${daily_stats.total_cost_usd:.2f} / "
                    f"${self.budget.daily_limit_usd:.2f} ({daily_pct*100:.0f}%)"
                )

        # Check session limit
        if self.budget.session_limit_usd and self.budget.session_limit_usd > 0:
            session_pct = session_stats.total_cost_usd / self.budget.session_limit_usd

            if session_pct >= 1.0:
                result["within_budget"] = False
                result["warning"] = (
                    f"Session budget exceeded: ${session_stats.total_cost_usd:.2f} / "
                    f"${self.budget.session_limit_usd:.2f}"
                )
            elif (
                session_pct >= self.budget.warning_threshold
                and not result["warning"]
            ):
                result["warning"] = (
                    f"Approaching session budget: ${session_stats.total_cost_usd:.2f} / "
                    f"${self.budget.session_limit_usd:.2f} ({session_pct*100:.0f}%)"
                )

        return result

    def get_cost_summary(self) -> dict[str, Any]:
        """Get comprehensive cost summary."""
        session_stats = self.get_session_stats()
        daily_stats = self.get_daily_stats()
        budget_status = self.check_budget()

        # Model breakdown for today
        model_usage: dict[str, dict[str, Any]] = {}
        for record in self._today_records:
            if record.model not in model_usage:
                model_usage[record.model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0,
                    "requests": 0,
                }
            model_usage[record.model]["input_tokens"] += record.input_tokens
            model_usage[record.model]["output_tokens"] += record.output_tokens
            model_usage[record.model]["cost_usd"] += record.cost_usd
            model_usage[record.model]["requests"] += 1

        return {
            "session": session_stats.to_dict(),
            "today": daily_stats.to_dict(),
            "budget": budget_status,
            "model_breakdown": model_usage,
            "session_duration_minutes": round(
                (time.time() - self._session_start) / 60, 1
            ),
        }

    def format_summary(self) -> str:
        """Get formatted cost summary string."""
        summary = self.get_cost_summary()
        session = summary["session"]
        today = summary["today"]
        budget = summary["budget"]

        lines = [
            "=== Cost Summary ===",
            "",
            "Session:",
            f"  Tokens: {session['total_tokens']:,} "
            f"(in: {session['total_input_tokens']:,}, out: {session['total_output_tokens']:,})",
            f"  Cost: ${session['total_cost_usd']:.4f}",
            f"  Requests: {session['request_count']}",
            "",
            "Today:",
            f"  Tokens: {today['total_tokens']:,}",
            f"  Cost: ${today['total_cost_usd']:.4f}",
            f"  Requests: {today['request_count']}",
        ]

        if budget["warning"]:
            lines.extend(["", f"⚠️ {budget['warning']}"])

        return "\n".join(lines)
