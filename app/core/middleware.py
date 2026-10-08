import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings

logger = structlog.get_logger("request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attaches a request ID to every request/response and logs timing.

    This is the backbone of docs/architecture.md section 29 (observability):
    every log line for a request can be correlated by request_id.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        header_name = settings.request_id_header
        request_id = request.headers.get(header_name, str(uuid.uuid4()))

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers[header_name] = request_id
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response
