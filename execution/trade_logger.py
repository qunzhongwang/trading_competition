"""Structured trade logger for competition compliance.

Writes append-only JSONL files for orders, signals, and API events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TradeLogger:
    """Thread-safe append-only JSONL logger for trade events."""

    def __init__(self, log_dir: str = "logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        self._date_str = datetime.utcnow().strftime("%Y%m%d")
        self._file_path = self._log_dir / f"trades_{self._date_str}.jsonl"

    def _rotate_if_needed(self) -> None:
        """Rotate log file if the date has changed."""
        today = datetime.utcnow().strftime("%Y%m%d")
        if today != self._date_str:
            self._date_str = today
            self._file_path = self._log_dir / f"trades_{self._date_str}.jsonl"

    async def _write_event(self, event: Dict[str, Any]) -> None:
        """Write a single event to the JSONL file."""
        self._rotate_if_needed()
        async with self._lock:
            line = json.dumps(event, default=str) + "\n"
            with open(self._file_path, "a") as f:
                f.write(line)

    async def log_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        order_id: str,
        status: str,
        roostoo_response: Optional[Dict] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log an order event."""
        await self._write_event(
            {
                "event": "order",
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
                "status": status,
                "roostoo_response": roostoo_response,
                "latency_ms": latency_ms,
            }
        )

    async def log_signal(
        self,
        symbol: str,
        alpha_score: float,
        engine_type: str,
        action: str,
        reasoning: Optional[str] = None,
    ) -> None:
        """Log a signal event."""
        await self._write_event(
            {
                "event": "signal",
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "alpha_score": alpha_score,
                "engine_type": engine_type,
                "action": action,
                "reasoning": reasoning,
            }
        )

    async def log_api(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        response_code: Optional[int] = None,
        success: bool = True,
        error_msg: Optional[str] = None,
    ) -> None:
        """Log an API event (with secrets redacted from params)."""
        safe_params = _redact_secrets(params) if params else None
        await self._write_event(
            {
                "event": "api",
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": endpoint,
                "params": safe_params,
                "response_code": response_code,
                "success": success,
                "error_msg": error_msg,
            }
        )


def _redact_secrets(params: Dict) -> Dict:
    """Redact sensitive keys from params dict."""
    sensitive_keys = {
        "api_key",
        "api_secret",
        "secret",
        "password",
        "token",
        "RST-API-KEY",
        "MSG-SIGNATURE",
    }
    return {
        k: "***REDACTED***" if k.lower() in {s.lower() for s in sensitive_keys} else v
        for k, v in params.items()
    }
