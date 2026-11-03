"""Read-only checks for links owned by the main notes repository."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal
from urllib.parse import unquote, urlsplit

from .markdown import MarkdownDocument, parse_document
from .scope import RepositoryScope, absolute_path


@dataclass(frozen=True)
class Diagnostic:
    path: Path
    line: int
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


def _path_problem(root: Path, target: Path) -> tuple[str, str] | None:
    current = root
    for part in target.relative_to(root).parts:
        if not current.is_dir():
            return "LINK_MISSING", "Local target does not exist"
        names = {entry.name for entry in current.iterdir()}
        if part not in names:
            matching = next((name for name in names if name.casefold() == part.casefold()), None)
            if matching is not None:
                return "LINK_CASE", f"Path spelling must match {matching!r} exactly"
            return "LINK_MISSING", "Local target does not exist"
        current /= part
    return None


def check_links(
    scope: RepositoryScope, files: list[Path],
    documents: dict[Path, MarkdownDocument] | None = None,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if documents is None:
        documents = {}

    def read_document(path: Path) -> MarkdownDocument:
        if path not in documents:
            documents[path] = parse_document(path)
        return documents[path]

    for source in files:
        document = read_document(source)
        for link in document.links:
            if re.match(r"^[A-Za-z]:[/\\]", unquote(link.destination)):
                diagnostics.append(Diagnostic(
                    source, link.line, "LINK_BOUNDARY",
                    f"Machine-local absolute paths are not repository links: {link.destination}",
                ))
                continue
            destination = urlsplit(link.destination)
            if destination.scheme or destination.netloc:
                continue
            decoded_path = unquote(destination.path)
            if "\\" in decoded_path:
                diagnostics.append(Diagnostic(
                    source, link.line, "LINK_PATH",
                    f"Local URLs must use / separators on every platform: {link.destination}",
                ))
                continue
            if decoded_path.startswith("/"):
                target = absolute_path(scope.root / decoded_path.lstrip("/"))
            else:
                target = absolute_path(source.parent / decoded_path) if decoded_path else source
            boundary = scope.boundary(target)
            if boundary in {"submodule", "repository"}:
                diagnostics.append(Diagnostic(
                    source, link.line, f"LINK_{boundary.upper()}",
                    f"External {boundary} content is not verified by the main repository: {link.destination}",
                    "warning",
                ))
                continue
            if boundary and boundary != "template":
                diagnostics.append(Diagnostic(
                    source, link.line, "LINK_BOUNDARY",
                    f"Link crosses the {boundary} boundary: {link.destination}",
                ))
                continue
            problem = _path_problem(scope.root, target)
            if problem:
                code, message = problem
                diagnostics.append(Diagnostic(source, link.line, code, f"{message}: {link.destination}"))
                continue
            if destination.fragment:
                if boundary != "template" and target.is_file() and target.suffix.lower() == ".md":
                    target_document = read_document(target)
                    if unquote(destination.fragment) in target_document.anchors:
                        continue
                    uncertain = target_document.has_unverified_anchors
                else:
                    uncertain = True
                if uncertain:
                    diagnostics.append(Diagnostic(
                        source, link.line, "LINK_ANCHOR_UNVERIFIED",
                        "Anchor requires manual verification (complex heading or unscanned target): "
                        f"{link.destination}", "warning",
                    ))
                else:
                    diagnostics.append(Diagnostic(
                        source, link.line, "LINK_ANCHOR",
                        f"Markdown anchor does not exist: {link.destination}",
                    ))
    return diagnostics
