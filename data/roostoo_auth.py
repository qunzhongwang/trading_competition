"""Roostoo API authentication — HMAC SHA256 signing."""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class RoostooAuth:
    """HMAC SHA256 authentication for the Roostoo mock exchange API.

    Usage:
        auth = RoostooAuth(api_key="...", api_secret="...")
        headers, query = auth.sign({"pair": "BTC/USD", "timestamp": auth.get_timestamp()})
        # Use headers for request, query as body/params
    """

    MAX_TIME_DRIFT_MS = 60_000  # reject if server-local drift exceeds 60 s

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret

    def sign(self, params: dict) -> Tuple[Dict[str, str], str]:
        """Sign request parameters.

        1. Sort params alphabetically by key.
        2. Join as key1=val1&key2=val2&...
        3. HMAC SHA256 with secret key.
        4. Return (headers, query_string).
        """
        sorted_items = sorted(params.items(), key=lambda x: x[0])
        query_string = "&".join(f"{k}={v}" for k, v in sorted_items)

        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "RST-API-KEY": self._api_key,
            "MSG-SIGNATURE": signature,
        }
        return headers, query_string

    @staticmethod
    def get_timestamp() -> int:
        """Return 13-digit millisecond timestamp."""
        return int(time.time() * 1000)

    async def validate_server_time(self, base_url: str) -> bool:
        """Check server time drift is within acceptable range.

        Returns True if drift is acceptable, False otherwise.
        """
        import aiohttp

        url = f"{base_url}/v3/serverTime"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    server_time = int(data.get("serverTime", 0))
                    local_time = self.get_timestamp()
                    drift = abs(server_time - local_time)
                    if drift > self.MAX_TIME_DRIFT_MS:
                        logger.warning(
                            "Server time drift too large: %d ms (max %d ms)",
                            drift, self.MAX_TIME_DRIFT_MS,
                        )
                        return False
                    logger.info("Server time drift: %d ms (OK)", drift)
                    return True
        except Exception as e:
            logger.error("Failed to validate server time: %s", e)
            return False
