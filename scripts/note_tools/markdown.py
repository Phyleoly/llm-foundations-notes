"""Read note metadata through a Markdown parser, without rewriting the source."""

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Callable
import unicodedata

from markdown_it import MarkdownIt
from markdown_it.rules_inline import autolink, backtick, html_inline, image, link
from markdown_it.rules_inline.state_inline import StateInline
from markdown_it.token import Token


@dataclass(frozen=True)
class MarkdownLink:
    destination: str
    line: int


@dataclass(frozen=True)
class MarkdownDocument:
    title: str | None
    h1_count: int
    statuses: list[tuple[str, int]]
    links: list[MarkdownLink]
    anchors: frozenset[str]
    has_unverified_anchors: bool


def _plain_text(tokens: list[Token]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token.type in {"text", "code_inline"}:
            parts.append(token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
    return "".join(parts)


class _ExplicitAnchors(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: set[str] = set()
        self.has_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.has_heading = True
        for name, value in attrs:
            if value is not None and (name == "id" or (tag == "a" and name == "name")):
                self.anchors.add(value)


def _heading_slug(title: str, used: set[str]) -> str:
    """GitHub's documented rules for ordinary headings, excluding complex markup."""
    base = "".join(
        character for character in title.lower()
        if character in {" ", "-", "_"}
        or unicodedata.category(character)[0] in {"L", "N", "M"}
    ).replace(" ", "-")
    candidate, suffix = base, 0
    while candidate in used:
        suffix += 1
        candidate = f"{base}-{suffix}"
    used.add(candidate)
    return candidate


def _with_source_line(
    rule: Callable[[StateInline, bool], bool], token_type: str
) -> Callable[[StateInline, bool], bool]:
    """Attach locations at the parser's own recognition point, including references."""
    def located_rule(state: StateInline, silent: bool) -> bool:
        position, token_count = state.pos, len(state.tokens)
        matched = rule(state, silent)
        if matched and not silent:
            for token in state.tokens[token_count:]:
                if token.type == token_type:
                    token.meta.setdefault("note_line_offset", state.src.count("\n", 0, position))
                    token.meta.setdefault("note_source_range", (position, state.pos))
                    break
        return matched

    return located_rule


def _parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark", {"html": True})
    parser.inline.ruler.at("link", _with_source_line(link, "link_open"))
    parser.inline.ruler.at("image", _with_source_line(image, "image"))
    parser.inline.ruler.at("autolink", _with_source_line(autolink, "link_open"))
    parser.inline.ruler.at("backticks", _with_source_line(backtick, "code_inline"))
    parser.inline.ruler.at("html_inline", _with_source_line(html_inline, "html_inline"))
    return parser


def _complex_heading(children: list[Token]) -> bool:
    return any(
        child.type in {"html_inline", "image"}
        or (child.type == "text" and "$" in child.content)
        for child in children
    )


def _metadata_content(token: Token) -> str:
    characters = list(token.content)
    for child in token.children or []:
        if child.type in {"code_inline", "html_inline"}:
            start, end = child.meta.get("note_source_range", (0, 0))
            for position in range(start, end):
                if characters[position] != "\n":
                    characters[position] = " "
    return "".join(characters)


def parse_document(path: Path) -> MarkdownDocument:
    return parse_markdown(path.read_text(encoding="utf-8"))


def parse_markdown(source: str) -> MarkdownDocument:
    tokens = _parser().parse(source)
    # Only consecutive top-level quotes after the first H1 are metadata.
    # Once prose starts, a quoted status is ordinary article content.
    header_start = next((index + 1 for index, token in enumerate(tokens)
                         if token.type == "heading_close" and token.tag == "h1"
                         and token.level == 0), len(tokens))
    header_end = next((index for index in range(header_start, len(tokens))
                       if tokens[index].level == 0
                       and tokens[index].type not in {"blockquote_open", "blockquote_close"}), len(tokens))
    titles: list[str] = []
    statuses: list[tuple[str, int]] = []
    links: list[MarkdownLink] = []
    heading_anchors: set[str] = set()
    used_slugs: set[str] = set()
    has_unverified_anchors = False
    explicit_anchors = _ExplicitAnchors()
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            children = tokens[index + 1].children or []
            title = _plain_text(children)
            slug = _heading_slug(title, used_slugs)
            if _complex_heading(children):
                has_unverified_anchors = True
            else:
                heading_anchors.add(slug)
            if token.tag == "h1":
                titles.append(title)
        elif token.type == "html_block":
            explicit_anchors.feed(token.content)
        elif token.type == "inline":
            first_line = (token.map or [0])[0] + 1
            for child in token.children or []:
                if child.type == "html_inline":
                    explicit_anchors.feed(child.content)
                attribute = {"link_open": "href", "image": "src"}.get(child.type)
                if attribute:
                    destination = child.attrGet(attribute)
                    if destination is not None:
                        links.append(MarkdownLink(
                            destination, first_line + child.meta.get("note_line_offset", 0)
                        ))
            if (header_start <= index < header_end and token.level == 2
                    and tokens[index - 1].type == "paragraph_open"):
                for offset, line in enumerate(_metadata_content(token).splitlines()):
                    match = re.fullmatch(r"状态[：:]\s*(.+?)\s*", line)
                    if match:
                        statuses.append((match.group(1), first_line + offset))
    return MarkdownDocument(
        titles[0] if titles else None, len(titles), statuses, links,
        frozenset(heading_anchors | explicit_anchors.anchors),
        has_unverified_anchors or explicit_anchors.has_heading,
    )
