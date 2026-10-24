"""Bounded file operations beneath an owned workspace.

On POSIX, directory descriptors keep path resolution independent of concurrent
renames and O_NOFOLLOW rejects links. Windows also rejects reparse points; the
sandbox must be stopped before controller access on either platform.
"""

from __future__ import annotations

import os
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path

from code_agent.sensitive_paths import relative_parts as relative_parts


def is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


@contextmanager
def parent_directory(root: Path, relative: str, *, create: bool = False):
    parts = relative_parts(relative)
    if os.name != "posix":
        directory = root
        for part in ("", *parts[:-1]):
            directory = directory / part
            if create:
                directory.mkdir(exist_ok=True)
            if is_link(directory.lstat()) or not directory.is_dir():
                raise ValueError("workspace paths cannot traverse links")
        yield directory, parts[-1], {}
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o777, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield None, parts[-1], {"dir_fd": descriptor}
    finally:
        os.close(descriptor)


def read_bytes(root: Path, relative: str, limit: int) -> bytes:
    with parent_directory(root, relative) as (parent, name, options):
        target = parent / name if parent is not None else name
        before = os.stat(target, follow_symlinks=False, **options)
        if is_link(before) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("only unlinked regular workspace files may be read")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(target, flags, **options)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                    or opened.st_size > limit):
                raise ValueError("file changed or exceeds workspace limits")
            data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError("file exceeds workspace limits")
            return data


def atomic_write(root: Path, relative: str, data: bytes) -> None:
    with parent_directory(root, relative, create=True) as (parent, name, options):
        temporary = f".agentforge-write-{secrets.token_hex(16)}"
        target = parent / name if parent is not None else name
        temp_target = parent / temporary if parent is not None else temporary
        descriptor = os.open(temp_target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600, **options)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                if os.name == "posix":
                    os.fchmod(stream.fileno(), 0o666)
            if options:
                os.replace(temp_target, target, src_dir_fd=options["dir_fd"], dst_dir_fd=options["dir_fd"])
            else:
                os.replace(temp_target, target)
        finally:
            try:
                os.unlink(temp_target, **options)
            except FileNotFoundError:
                pass


def unlink(root: Path, relative: str) -> None:
    with parent_directory(root, relative) as (parent, name, options):
        target = parent / name if parent is not None else name
        info = os.stat(target, follow_symlinks=False, **options)
        if is_link(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError("only regular workspace files can be deleted")
        os.unlink(target, **options)
