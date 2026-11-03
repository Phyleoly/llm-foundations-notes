#!/usr/bin/env python3
"""Create and maintain ordinary Markdown notes; validation never writes."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING
from urllib.parse import quote

from note_tools.scope import RepositoryScope

if TYPE_CHECKING:
    from note_tools.markdown import MarkdownDocument

START = '<!-- notes:index:start -->'
END = '<!-- notes:index:end -->'
STATUSES = frozenset({'提纲', '草稿', '已核对', '已归档'})
NOTE_TEMPLATES = ('note', 'paper_reading')


def read_source(path: Path) -> str:
    with path.open(encoding='utf-8', newline='') as handle:
        return handle.read()


def note_errors(document: MarkdownDocument) -> list[tuple[int, str, str]]:
    issues = []
    if document.h1_count != 1 or not document.title:
        issues.append((1, 'NOTE001', 'article requires exactly one nonempty H1'))
    if len(document.statuses) != 1:
        issues.append((1, 'NOTE002', 'article requires one visible > 状态：... line immediately after H1'))
    elif document.statuses[0][0] not in STATUSES:
        issues.append((document.statuses[0][1], 'NOTE003', 'unknown article status'))
    return issues


def index_changes(
    scope: RepositoryScope, notes: dict[Path, MarkdownDocument] | None = None,
) -> list[tuple[Path, str]]:
    from markdown_it import MarkdownIt
    from note_tools.markdown import parse_document

    if notes is None:
        notes = {path: parse_document(path) for path in scope.note_files()}
    for path, document in notes.items():
        errors = note_errors(document)
        if errors:
            line, code, message = errors[0]
            raise ValueError(f'{path.relative_to(scope.root)}:{line}: {code}: {message}')
    changes = []
    # Prepare every change before writing anything: a malformed later chapter
    # must not leave earlier chapter inventories partially updated.
    for chapter in scope.chapters():
        path = chapter / 'README.md'
        if scope.boundary(path) is not None:
            raise ValueError(f'{path}: chapter inventory must be a regular main-repository file')
        if 'README.md' not in {entry.name for entry in chapter.iterdir()}:
            raise ValueError(f'{chapter}: expected README.md with this exact spelling')
        source = read_source(path)
        markers = [(token.content.strip(), token.map[0])
                   for token in MarkdownIt('commonmark').parse(source)
                   if token.type == 'html_block' and token.level == 0
                   and token.content.strip() in {START, END}]
        if [marker for marker, _ in markers] != [START, END]:
            raise ValueError(f'{path}: expected exactly one index marker pair')
        lines = source.splitlines(keepends=True)
        start_line, end_line = markers[0][1], markers[1][1]
        if lines[start_line].strip() != START or lines[end_line].strip() != END:
            raise ValueError(f'{path}: index markers must be standalone lines')
        start = sum(map(len, lines[:start_line])) + lines[start_line].index(START)
        end = sum(map(len, lines[:end_line]))
        newline = '\r\n' if '\r\n' in source else '\n'
        rows = ['', '| 笔记 | 状态 |', '| --- | --- |']
        for note, document in notes.items():
            if scope.chapter_for(note) != chapter:
                continue
            # Titles are plain text here; preserve code characters as visible
            # text rather than reinterpreting them as HTML, Markdown or math.
            title = re.sub(r'([\\`*_~\[\]|$])', r'\\\1', escape(document.title, quote=False))
            target = quote(note.relative_to(chapter).as_posix(), safe='/')
            rows.append(f'| [{title}](./{target}) | {document.statuses[0][0]} |')
        rows.append('')
        generated = source[:start + len(START)] + newline.join(rows) + newline + source[end:]
        if generated != source:
            changes.append((path, generated))
    return changes


def update_index(scope: RepositoryScope, check: bool = False) -> int:
    changes = index_changes(scope)
    for path, source in changes:
        relative = path.relative_to(scope.root).as_posix()
        if check:
            print(f'{relative}:1: INDEX001: inventory is stale; run index')
        else:
            with path.open('w', encoding='utf-8', newline='') as handle:
                handle.write(source)
            print(f'Updated {relative}')
    if not changes:
        print('Inventories are up to date.')
    return 1 if check and changes else 0


def check_repository(scope: RepositoryScope) -> int:
    import check_markdown_math
    from note_tools.links import check_links
    from note_tools.markdown import parse_document

    files = scope.markdown_files()
    documents = {path: parse_document(path) for path in files}
    errors = warnings = 0
    notes = {path: document for path, document in documents.items() if scope.is_note(path)}
    for path, document in notes.items():
        for line, code, message in note_errors(document):
            print(f'{path.relative_to(scope.root).as_posix()}:{line}: {code}: {message}')
            errors += 1
    for issue in check_links(scope, files, documents):
        print(f'{issue.path.relative_to(scope.root).as_posix()}:{issue.line}: '
              f'{issue.code} ({issue.severity}): {issue.message}')
        if issue.severity == 'warning':
            warnings += 1
        else:
            errors += 1
    for path in files:
        relative = path.relative_to(scope.root).as_posix()
        analysis = check_markdown_math.analyze_document(read_source(path), relative)
        for issue in analysis.issues:
            print(issue.render())
            errors += 1
    try:
        for path, _ in index_changes(scope, notes):
            print(f'{path.relative_to(scope.root).as_posix()}:1: INDEX001: inventory is stale; run index')
            errors += 1
    except ValueError as error:
        print(f'INDEX002: {error}')
        errors += 1
    print(f'Checked {len(files)} Markdown files, {len(notes)} articles; '
          f'{errors} error(s), {warnings} unverified target(s).')
    return 1 if errors else 0


def new_note(
    scope: RepositoryScope, relative_path: str, title: str, template_name: str = 'note',
) -> Path:
    from note_tools.markdown import parse_markdown

    relative = Path(relative_path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('new requires a repository-relative article path without ..')
    target = scope.root / relative
    if not scope.is_note(target) or not re.fullmatch(r'[a-z][a-z0-9_]*\.md', target.name):
        raise ValueError('use a chapter path and an unnumbered snake_case.md filename')
    chapter = scope.chapter_for(target)
    if chapter is None or not chapter.is_dir():
        raise ValueError('create the chapter and its reading guide before adding notes')
    title = title.strip()
    if not title or '\n' in title or '\r' in title:
        raise ValueError('title must be a nonempty single line')
    if template_name not in NOTE_TEMPLATES:
        raise ValueError(f'unknown article template: {template_name}')
    template = (scope.root / 'templates' / f'{template_name}.md').read_text(encoding='utf-8')
    if template.count('{{ title }}') != 1:
        raise ValueError('the shared template must contain exactly one {{ title }} placeholder')
    source = template.replace('{{ title }}', title)
    document = parse_markdown(source)
    errors = note_errors(document)
    if errors:
        raise ValueError(f'the rendered template is invalid: {errors[0][2]}')
    if document.statuses[0][0] != '草稿':
        raise ValueError('the shared template must create a draft (状态：草稿)')
    # Check case collisions even on case-sensitive systems, for portable paths.
    current = scope.root
    for part in relative.parts:
        if current.is_dir():
            conflicts = [p.name for p in current.iterdir()
                         if p.name.casefold() == part.casefold() and p.name != part]
            if conflicts:
                raise ValueError(f'path casing conflicts with {conflicts[0]}')
        current = current / part
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(source)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='repository root (default: this script\'s repository)')
    commands = parser.add_subparsers(dest='command', required=True)
    new = commands.add_parser('new', help='create a draft without overwriting files')
    new.add_argument('path')
    new.add_argument('--title', required=True)
    new.add_argument('--template', choices=NOTE_TEMPLATES, default='note',
                     help='article template (default: note)')
    index = commands.add_parser('index', help='update generated chapter inventories only')
    index.add_argument('--check', action='store_true', help='report stale inventories without writing')
    commands.add_parser('check', help='validate notes, links, inventories and math without writing')
    args = parser.parse_args(argv)
    try:
        scope = RepositoryScope(args.root)
        if args.command == 'new':
            path = new_note(scope, args.path, args.title, args.template)
            print(f'Created {path.relative_to(scope.root).as_posix()}')
        elif args.command == 'index':
            return update_index(scope, args.check)
        elif args.command == 'check':
            return check_repository(scope)
        return 0
    except (OSError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
