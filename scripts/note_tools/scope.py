"""One filesystem boundary for note discovery, validation and math fixes.

Discovery never needs Git's index or a checked-out submodule. This deliberately
includes newly written, untracked articles while pruning external worktrees.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess


EXCLUDED_DIRECTORIES = frozenset({
    '.git', '.venv', 'venv', '.mypy_cache', '.pytest_cache', '.ruff_cache',
    '.cache', '__pycache__', '.ipynb_checkpoints', 'node_modules', 'vendor',
    '_backups', '.tmp-tests',
})
CHAPTER_NAME = re.compile(r'\d{2}_[a-z][a-z0-9_]*\Z')
NON_ARTICLE_DIRECTORIES = frozenset({'code', 'src', 'tests', 'assets'})


def absolute_path(path: Path) -> Path:
    """Normalize dot segments without following a symlink across a boundary."""
    return Path(os.path.abspath(path))


def is_link(path: Path) -> bool:
    """Also exclude Windows junctions when running Python 3.11."""
    if path.is_symlink():
        return True
    if os.name == 'nt':
        try:
            attributes = path.stat(follow_symlinks=False).st_file_attributes
        except FileNotFoundError:
            return False
        return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return False


def repository_root(path: Path) -> Path:
    """Find the owning repository; standalone files retain their local scope."""
    start = absolute_path(path if path.is_dir() else path.parent)
    target = absolute_path(path)
    nearest = None
    for candidate in (start, *start.parents):
        has_modules = (candidate / '.gitmodules').is_file()
        if has_modules:
            # An initialized module has its own .git marker. Explicit input
            # must not use that marker to bypass its parent's declaration.
            scope = RepositoryScope(candidate)
            if any(target == module or module in target.parents for module in scope.submodule_paths):
                return candidate
        if nearest is None and ((candidate / '.git').exists() or has_modules):
            nearest = candidate
    return nearest or start


class RepositoryScope:
    def __init__(self, root: Path):
        self.root = absolute_path(root)
        if any(is_link(candidate) for candidate in (self.root, *self.root.parents)):
            raise ValueError(f'{self.root}: repository root must not pass through a symlink or junction')
        self.submodule_paths = self._read_submodule_paths()

    def _read_submodule_paths(self) -> list[Path]:
        config_path = self.root / '.gitmodules'
        if not config_path.is_file():
            return []
        if is_link(config_path):
            raise ValueError(f'{config_path}: submodule configuration must be a regular file')
        try:
            result = subprocess.run(
                ['git', 'config', '--no-includes', '--file', str(config_path),
                 '--null', '--get-regexp', r'^submodule\.'],
                capture_output=True, check=False,
            )
        except FileNotFoundError as error:
            raise ValueError('Git is required to read .gitmodules safely; no network is used') from error
        if result.returncode not in (0, 1):
            raise ValueError(f'{config_path}: invalid Git configuration: {result.stderr.decode("utf-8", errors="replace").strip()}')
        sections: dict[str, list[str]] = {}
        for entry in result.stdout.decode('utf-8').split('\0'):
            if not entry:
                continue
            key, _, value = entry.partition('\n')
            section, _, field = key.rpartition('.')
            sections.setdefault(section, [])
            if field == 'path':
                sections[section].append(value)
        result = []
        for section, paths in sections.items():
            if len(paths) != 1:
                raise ValueError(f'{config_path}: expected one path in {section}')
            value = paths[0]
            path = PurePosixPath(value)
            if (not value or path.is_absolute() or '..' in path.parts
                    or '\\' in value or ':' in value or str(path) == '.'
                    or any(ord(character) < 32 for character in value)):
                raise ValueError(f'{config_path}: invalid submodule path in {section}')
            result.append(self.root.joinpath(*path.parts))
        return sorted(set(result))

    def boundary(self, path: Path) -> str | None:
        path = absolute_path(path)
        try:
            parts = path.relative_to(self.root).parts
        except ValueError:
            return 'outside'
        # A nested repository must not hide the reserved backup boundary.
        if any(part.casefold() == '_backups' for part in path.parts):
            return 'backup'
        if any(path == module or module in path.parents for module in self.submodule_paths):
            return 'submodule'
        if any(part.casefold() in EXCLUDED_DIRECTORIES for part in parts):
            return 'excluded'
        current = self.root
        for part in parts:
            current = current / part
            if is_link(current):
                return 'excluded'
            if current.is_dir() and (current / '.git').exists():
                return 'repository'
        if any(part.casefold() == 'templates' for part in parts):
            return 'template'
        return None

    def markdown_files(self, start: Path | None = None) -> list[Path]:
        start = absolute_path(start or self.root)
        if self.boundary(start):
            return []
        if start.is_file():
            return [start] if start.suffix.lower() == '.md' else []
        if not start.is_dir():
            raise FileNotFoundError(start)
        result = []
        for directory, names, files in os.walk(start, followlinks=False):
            parent = Path(directory)
            names[:] = sorted(name for name in names if self.boundary(parent / name) is None)
            for name in sorted(files):
                path = parent / name
                if path.suffix.lower() == '.md' and self.boundary(path) is None:
                    result.append(path)
        return sorted(result, key=lambda path: path.relative_to(self.root).as_posix())

    def chapter_for(self, path: Path) -> Path | None:
        try:
            parts = absolute_path(path).relative_to(self.root).parts
        except ValueError:
            return None
        if parts and CHAPTER_NAME.fullmatch(parts[0]):
            return self.root / parts[0]
        return None

    def is_note(self, path: Path) -> bool:
        chapter = self.chapter_for(path)
        if chapter is None or self.boundary(path):
            return False
        relative = absolute_path(path).relative_to(chapter)
        return (path.suffix.lower() == '.md' and path.name.casefold() != 'readme.md'
                and not any(part.casefold() in NON_ARTICLE_DIRECTORIES for part in relative.parts[:-1]))

    def note_files(self) -> list[Path]:
        return [path for path in self.markdown_files() if self.is_note(path)]

    def chapters(self) -> list[Path]:
        return sorted(path for path in self.root.iterdir()
                      if path.is_dir() and CHAPTER_NAME.fullmatch(path.name)
                      and self.boundary(path) is None)
