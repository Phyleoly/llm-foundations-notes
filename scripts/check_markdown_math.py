#!/usr/bin/env python3
"""Check and conservatively repair GitHub Markdown math.

The scanner is context-aware: it ignores ordinary fenced code blocks, inline
code spans, HTML comments, and raw HTML code-like blocks.  Automatic fixes are
limited to deterministic transformations:

* Convert ``$$`` blocks containing a standalone ``=`` or ``-`` line to a
  GitHub-supported ``math`` fence so GFM cannot parse the line as a Setext
  heading.
* Replace a small allowlist of ``\\operatorname{name}`` uses with
  ``\\mathrm{name}``.  Unknown operator names are reported but never changed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence


FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
DISPLAY_DELIMITER_RE = re.compile(r"^(?P<indent> {0,3})\$\$[ \t]*$")
SETEXT_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
OPERATOR_RE = re.compile(
    r"\\operatorname(?P<star>\*)?[ \t]*\{(?P<name>[^{}]+)\}"
)
ENVIRONMENT_RE = re.compile(r"\\(?P<kind>begin|end)\{(?P<name>[^{}]+)\}")
RAW_HTML_OPEN_RE = re.compile(r"<(pre|script|style|textarea)\b", re.IGNORECASE)
INLINE_CODE_HTML_RE = re.compile(
    r"<code\b[^>]*>.*?</code[ \t]*>", re.IGNORECASE
)
INLINE_COMMENT_RE = re.compile(r"<!--.*?-->")

SAFE_OPERATOR_NAMES = frozenset(
    {
        "Categorical",
        "Lik",
        "LSE",
        "Softmax",
        "id",
        "logsoftmax",
        "logsumexp",
        "softmax",
    }
)

EXCLUDED_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".venv", "node_modules", "vendor"}
)


@dataclass(frozen=True)
class Issue:
    path: str
    line: int
    code: str
    message: str
    fixable: bool = False

    def render(self) -> str:
        suffix = " (run with --fix)" if self.fixable else ""
        return f"{self.path}:{self.line}: {self.code} {self.message}{suffix}"


@dataclass
class Analysis:
    issues: list[Issue]
    text: str
    fix_count: int = 0


@dataclass
class Fence:
    character: str
    length: int
    info: str
    start_line: int
    start_index: int


@dataclass
class MathBlock:
    kind: str
    start_line: int
    start_index: int
    body: list[tuple[int, int, str]]


def _split_line(raw_line: str) -> tuple[str, str]:
    if raw_line.endswith("\r\n"):
        return raw_line[:-2], "\r\n"
    if raw_line.endswith("\n") or raw_line.endswith("\r"):
        return raw_line[:-1], raw_line[-1]
    return raw_line, ""


def _with_original_ending(content: str, raw_line: str) -> str:
    _, ending = _split_line(raw_line)
    return content + ending


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _strip_tex_comment(line: str) -> str:
    for index, character in enumerate(line):
        if character == "%" and not _is_escaped(line, index):
            return line[:index]
    return line


def _mask_range(characters: list[str], start: int, end: int) -> None:
    for index in range(start, end):
        characters[index] = " "


def _mask_inline_protected_regions(line: str) -> str:
    """Mask code spans and inline HTML without changing character offsets."""

    characters = list(line)
    index = 0
    while index < len(line):
        if line[index] != "`":
            index += 1
            continue
        end_ticks = index
        while end_ticks < len(line) and line[end_ticks] == "`":
            end_ticks += 1
        delimiter = line[index:end_ticks]
        closing = line.find(delimiter, end_ticks)
        if closing < 0:
            index = end_ticks
            continue
        end = closing + len(delimiter)
        _mask_range(characters, index, end)
        index = end

    for pattern in (INLINE_CODE_HTML_RE, INLINE_COMMENT_RE):
        for match in pattern.finditer(line):
            _mask_range(characters, match.start(), match.end())

    return "".join(characters)


def _inline_math_spans(line: str) -> tuple[list[tuple[int, int]], int | None]:
    """Return likely inline-math spans and an unmatched opening delimiter.

    Delimiters inside code/HTML are masked.  Dollar amounts such as ``$5`` are
    not treated as math unless a closing dollar appears before whitespace,
    which keeps the checker conservative around ordinary prose.
    """

    masked = _mask_inline_protected_regions(line)
    positions = [
        index
        for index, character in enumerate(masked)
        if character == "$"
        and not _is_escaped(masked, index)
        and (index == 0 or masked[index - 1] != "$")
        and (index + 1 == len(masked) or masked[index + 1] != "$")
    ]

    spans: list[tuple[int, int]] = []
    opening: int | None = None
    for position_index, position in enumerate(positions):
        previous = masked[position - 1] if position else ""
        following = masked[position + 1] if position + 1 < len(masked) else ""
        can_open = bool(following and not following.isspace())
        can_close = bool(previous and not previous.isspace())

        if opening is None:
            if not can_open:
                continue
            if following.isdigit():
                closes_before_space = any(
                    not any(character.isspace() for character in masked[position + 1 : later])
                    for later in positions[position_index + 1 :]
                    if later > position
                )
                if not closes_before_space:
                    continue
            opening = position
        elif can_close:
            spans.append((opening, position + 1))
            opening = None

    return spans, opening


def _operator_issues_and_replacement(
    tex: str,
    *,
    path: str,
    line: int,
    apply_fixes: bool,
) -> tuple[list[Issue], str, int]:
    issues: list[Issue] = []
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        name = match.group("name")
        star = match.group("star")
        fixable = star is None and name in SAFE_OPERATOR_NAMES
        issues.append(
            Issue(
                path,
                line,
                "MATH003",
                f"GitHub-incompatible \\operatorname{{{name}}}; use an allowed upright form",
                fixable=fixable,
            )
        )
        if apply_fixes and fixable:
            replacements += 1
            return rf"\mathrm{{{name}}}"
        return match.group(0)

    return issues, OPERATOR_RE.sub(replace, tex), replacements


def _latex_structure_issues(
    body: Sequence[tuple[int, int, str]], path: str, fallback_line: int
) -> list[Issue]:
    issues: list[Issue] = []
    braces: list[int] = []
    environments: list[tuple[str, int]] = []

    for _, line_number, original in body:
        line = _strip_tex_comment(original)
        for index, character in enumerate(line):
            if _is_escaped(line, index):
                continue
            if character == "{":
                braces.append(line_number)
            elif character == "}":
                if braces:
                    braces.pop()
                else:
                    issues.append(
                        Issue(path, line_number, "MATH005", "unmatched closing brace")
                    )

        for match in ENVIRONMENT_RE.finditer(line):
            kind = match.group("kind")
            name = match.group("name")
            if kind == "begin":
                environments.append((name, line_number))
            elif environments and environments[-1][0] == name:
                environments.pop()
            else:
                issues.append(
                    Issue(
                        path,
                        line_number,
                        "MATH006",
                        f"unmatched \\end{{{name}}}",
                    )
                )

    if braces:
        issues.append(
            Issue(path, braces[0], "MATH005", "unclosed opening brace in math")
        )
    for name, line_number in environments:
        issues.append(
            Issue(
                path,
                line_number or fallback_line,
                "MATH006",
                f"unclosed \\begin{{{name}}}",
            )
        )
    return issues


def analyze_document(text: str, path: str = "<text>", apply_fixes: bool = False) -> Analysis:
    lines = text.splitlines(keepends=True)
    issues: list[Issue] = []
    replacements: dict[int, str] = {}
    fix_count = 0
    fence: Fence | None = None
    math_block: MathBlock | None = None
    html_closing_tag: str | None = None

    def inspect_math(block: MathBlock, closing_index: int | None) -> None:
        nonlocal fix_count
        risky_line = next(
            (
                line_number
                for _, line_number, content in block.body
                if SETEXT_RE.fullmatch(content)
            ),
            None,
        )
        if block.kind == "dollar" and risky_line is not None:
            issues.append(
                Issue(
                    path,
                    risky_line,
                    "MATH001",
                    "standalone '=' or '-' can turn this $$ block into a GFM Setext heading",
                    fixable=True,
                )
            )
            if apply_fixes and closing_index is not None:
                opening_content, _ = _split_line(lines[block.start_index])
                closing_content, _ = _split_line(lines[closing_index])
                opening_match = DISPLAY_DELIMITER_RE.fullmatch(opening_content)
                closing_match = DISPLAY_DELIMITER_RE.fullmatch(closing_content)
                if opening_match and closing_match:
                    replacements[block.start_index] = _with_original_ending(
                        opening_match.group("indent") + "```math",
                        lines[block.start_index],
                    )
                    replacements[closing_index] = _with_original_ending(
                        closing_match.group("indent") + "```", lines[closing_index]
                    )
                    fix_count += 1

        for index, line_number, content in block.body:
            operator_issues, replacement, count = _operator_issues_and_replacement(
                content,
                path=path,
                line=line_number,
                apply_fixes=apply_fixes,
            )
            issues.extend(operator_issues)
            if count:
                replacements[index] = _with_original_ending(replacement, lines[index])
                fix_count += count

        issues.extend(_latex_structure_issues(block.body, path, block.start_line))

    for index, raw_line in enumerate(lines):
        line_number = index + 1
        content, _ = _split_line(raw_line)
        lowered = content.lower()

        if fence is not None:
            closing_re = re.compile(
                r"^ {0,3}"
                + re.escape(fence.character)
                + "{"
                + str(fence.length)
                + r",}[ \t]*$"
            )
            if closing_re.fullmatch(content):
                if fence.info == "math":
                    assert math_block is not None
                    inspect_math(math_block, index)
                    math_block = None
                fence = None
            elif fence.info == "math":
                assert math_block is not None
                math_block.body.append((index, line_number, content))
            continue

        if math_block is not None:
            delimiter = DISPLAY_DELIMITER_RE.fullmatch(content)
            if math_block.kind == "dollar" and delimiter:
                inspect_math(math_block, index)
                math_block = None
            else:
                math_block.body.append((index, line_number, content))
            continue

        if html_closing_tag is not None:
            if html_closing_tag in lowered:
                html_closing_tag = None
            continue
        if "<!--" in content:
            if "-->" not in content[content.find("<!--") + 4 :]:
                html_closing_tag = "-->"
            continue
        raw_html_match = RAW_HTML_OPEN_RE.search(content)
        if raw_html_match:
            closing_tag = f"</{raw_html_match.group(1).lower()}>"
            if closing_tag not in lowered[raw_html_match.end() :]:
                html_closing_tag = closing_tag
            continue

        fence_match = FENCE_OPEN_RE.fullmatch(content)
        if fence_match:
            sequence = fence_match.group(2)
            info = fence_match.group(3).strip()
            fence = Fence(sequence[0], len(sequence), info, line_number, index)
            if info == "math":
                math_block = MathBlock("fence", line_number, index, [])
            continue

        if DISPLAY_DELIMITER_RE.fullmatch(content):
            math_block = MathBlock("dollar", line_number, index, [])
            continue

        spans, unmatched = _inline_math_spans(content)
        if unmatched is not None:
            issues.append(
                Issue(path, line_number, "MATH004", "unclosed inline-math delimiter")
            )

        updated_content = content
        inline_replacements: list[tuple[int, int, str]] = []
        for start, end in spans:
            tex = content[start + 1 : end - 1]
            operator_issues, replacement, count = _operator_issues_and_replacement(
                tex,
                path=path,
                line=line_number,
                apply_fixes=apply_fixes,
            )
            issues.extend(operator_issues)
            fix_count += count
            if count:
                inline_replacements.append((start + 1, end - 1, replacement))
            inline_body = [(index, line_number, replacement if count else tex)]
            issues.extend(_latex_structure_issues(inline_body, path, line_number))

        for start, end, replacement in reversed(inline_replacements):
            updated_content = updated_content[:start] + replacement + updated_content[end:]
        if updated_content != content:
            replacements[index] = _with_original_ending(updated_content, raw_line)

    if math_block is not None:
        if math_block.kind == "dollar":
            issues.append(
                Issue(path, math_block.start_line, "MATH002", "unclosed $$ math block")
            )
        else:
            issues.append(
                Issue(path, math_block.start_line, "MATH002", "unclosed math fence")
            )
    elif fence is not None:
        issues.append(
            Issue(path, fence.start_line, "MD001", "unclosed Markdown code fence")
        )
    if html_closing_tag == "-->":
        issues.append(Issue(path, len(lines) or 1, "MD002", "unclosed HTML comment"))

    if replacements:
        for index, replacement in replacements.items():
            lines[index] = replacement
    return Analysis(issues, "".join(lines), fix_count)


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRECTORIES for part in path.parts)


def find_markdown_files(inputs: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_file():
            if path.suffix.lower() == ".md" and not _is_excluded(path):
                files.add(path)
            continue
        if path.is_dir():
            for candidate in path.rglob("*.md"):
                if (
                    candidate.is_file()
                    and not candidate.is_symlink()
                    and not _is_excluded(candidate)
                ):
                    files.add(candidate)
            continue
        raise FileNotFoundError(value)
    return sorted(files, key=lambda item: item.as_posix())


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_preserving_newlines(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def process_file(path: Path, fix: bool = False) -> tuple[list[Issue], bool, int]:
    display_path = _display_path(path)
    original = _read_preserving_newlines(path)
    first_pass = analyze_document(original, display_path, apply_fixes=fix)
    changed = fix and first_pass.text != original
    if changed:
        _write_preserving_newlines(path, first_pass.text)
        final = analyze_document(first_pass.text, display_path, apply_fixes=False)
        return final.issues, True, first_pass.fix_count
    return first_pass.issues, False, first_pass.fix_count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Markdown math for GitHub rendering hazards."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Markdown files or directories to scan (default: current directory)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply only deterministic, allowlisted fixes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        files = find_markdown_files(args.paths)
    except FileNotFoundError as error:
        print(f"error: path does not exist: {error}", file=sys.stderr)
        return 2

    if not files:
        print("error: no Markdown files found", file=sys.stderr)
        return 2

    all_issues: list[Issue] = []
    changed_files = 0
    applied_fixes = 0
    for path in files:
        issues, changed, fix_count = process_file(path, fix=args.fix)
        all_issues.extend(issues)
        changed_files += int(changed)
        applied_fixes += fix_count

    all_issues.sort(key=lambda issue: (issue.path, issue.line, issue.code))
    for issue in all_issues:
        print(issue.render())

    if args.fix:
        print(
            f"Applied {applied_fixes} fix(es) across {changed_files} file(s); "
            f"{len(all_issues)} issue(s) remain."
        )
    else:
        print(f"Checked {len(files)} Markdown file(s); found {len(all_issues)} issue(s).")
    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
