"""Operator-owned repository grants; absent configuration grants no access."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from code_agent.safe_files import is_link, relative_parts


class RepositoryAccess:
    def __init__(self, grants: Mapping[str, Sequence[str]] | None = None):
        if grants is None:
            raw = os.getenv("CODE_AGENT_REPOSITORY_GRANTS", "{}")
            if len(raw) > 64000:
                raise ValueError("repository grant configuration is too large")
            grants = json.loads(raw)
        if not isinstance(grants, Mapping) or len(grants) > 1000:
            raise ValueError("repository grants must map user IDs to repository names")
        self._grants = {}
        for owner, repositories in grants.items():
            if (not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9._-]{3,64}|\*", owner)
                    or not isinstance(repositories, (list, tuple)) or len(repositories) > 100
                    or any(not isinstance(item, str) for item in repositories)):
                raise ValueError("invalid repository grant")
            self._grants[owner] = frozenset("/".join(relative_parts(item)) for item in repositories)

    def allows(self, owner: str, repository: str) -> bool:
        name = "/".join(relative_parts(repository))
        return name in self._grants.get(owner, ()) or name in self._grants.get("*", ())

    def require(self, owner: str, repository: str, root: Path) -> None:
        if not owner or not self.allows(owner, repository):
            raise PermissionError("repository is not authorized for this user")
        path = root
        for part in relative_parts(repository):
            path = path / part
            if path.exists() and is_link(path.lstat()):
                raise PermissionError("repository grants cannot traverse links")
