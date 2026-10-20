"""Reported model-usage coverage, separate from token totals and provider billing."""

from threading import Lock

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage


class CountingUsageCallback(UsageMetadataCallbackHandler):
    def __init__(self):
        super().__init__()
        self.calls_with_usage = 0
        self.count_lock = Lock()

    def on_llm_end(self, response, **kwargs):
        super().on_llm_end(response, **kwargs)
        if not response.generations or not response.generations[0]:
            return
        message = getattr(response.generations[0][0], "message", None)
        if isinstance(message, AIMessage) and message.usage_metadata and message.response_metadata.get("model_name"):
            with self.count_lock:
                self.calls_with_usage += 1
