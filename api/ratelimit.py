"""Per-client request limit. A second layer behind nginx's limit_req, and the only one if nginx is bypassed."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

EXEMPT = {"/health"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, per_minute: int) -> None:
        super().__init__(app)
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._calls = 0

    async def dispatch(self, request, call_next):
        if request.url.path in EXEMPT:
            return await call_next(request)
        ip = request.client.host if request.client else "unknown"   # real client IP when uvicorn runs with --proxy-headers
        now = time.monotonic()
        with self._lock:
            q = self._hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return JSONResponse({"detail": "Too many requests. Slow down and retry in a few seconds."},
                                    status_code=429, headers={"Retry-After": "10"})
            q.append(now)
            self._calls += 1
            if self._calls % 2000 == 0:                              # forget clients that went quiet
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > 60]:
                    del self._hits[k]
        return await call_next(request)
