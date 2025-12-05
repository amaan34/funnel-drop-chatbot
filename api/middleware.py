import logging
import time
from typing import Callable

from fastapi import Request

logger = logging.getLogger("uvicorn.error")


async def logging_middleware(request: Request, call_next: Callable):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info("%s %s completed in %.1f ms", request.method, request.url.path, duration)
    return response

