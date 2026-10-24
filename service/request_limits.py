"""Bound actual ASGI request bytes, including chunked requests without a length."""

import asyncio

from starlette.responses import PlainTextResponse


class RequestBodyLimitMiddleware:
    def __init__(self, app, limit_for_scope, timeout_seconds=30):
        self.app = app
        self.limit_for_scope = limit_for_scope
        self.timeout_seconds = timeout_seconds

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH", "DELETE"}:
            return await self.app(scope, receive, send)
        limit = self.limit_for_scope(scope)
        body = bytearray()
        size = 0
        try:
            async with asyncio.timeout(self.timeout_seconds):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > limit:
                        return await PlainTextResponse("Request body too large", status_code=413)(scope, receive, send)
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return await PlainTextResponse("Request body timed out", status_code=408)(scope, receive, send)
        body = bytes(body)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)
