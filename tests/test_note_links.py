"""Public behavior of Markdown metadata and local-link checks."""

from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from note_tools.markdown import parse_document, parse_markdown
from note_tools.links import check_links
from note_tools.scope import RepositoryScope


class MarkdownDocumentTests(unittest.TestCase):
    def test_body_quotes_cannot_set_or_duplicate_article_status(self):
        for body in ['正文。\n', '## Section\n', '- Item\n', '```text\nexample\n```\n']:
            with self.subTest(body=body):
                quoted_status = '\n> 状态：已核对\n'
                self.assertEqual(parse_markdown('# Title\n\n' + body + quoted_status).statuses, [])
                source = '# Title\n\n> 状态：草稿\n\n' + body + quoted_status
                self.assertEqual(parse_markdown(source).statuses, [('草稿', 3)])

    def test_header_status_ignores_nested_quotes_and_lists_but_preserves_duplicates(self):
        document = parse_markdown(
            '# Title\n\n> 状态：草稿\n>\n> > 状态：已核对\n'
            '>\n> - 状态：已归档\n\n> 状态：提纲\n')
        self.assertEqual(document.statuses, [('草稿', 3), ('提纲', 9)])

    def test_metadata_comes_from_visible_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "note.md"
            path.write_text(
                "<!--\n# Hidden heading\n状态：已复核\n-->\n"
                "```markdown\n# Example heading\n状态：提纲\n```\n"
                "# **真正标题**\n\n> 状态：草稿\n",
                encoding="utf-8",
            )

            document = parse_document(path)

            self.assertEqual(document.title, "真正标题")
            self.assertEqual(document.h1_count, 1)
            self.assertEqual(document.statuses, [("草稿", 11)])

    def test_links_are_parsed_with_source_lines_and_skip_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "note.md"
            path.write_text(
                "# Note\n\n[first](target.md)\nsecond [reference][ref]\n\n"
                "![plot](assets/plot%20one.png)\n\n"
                "`[inline example](missing.md)`\n\n"
                "```md\n[fenced example](missing.md)\n```\n\n"
                "<!-- [comment example](missing.md) -->\n\n"
                "[ref]: target.md#section\n",
                encoding="utf-8",
            )

            document = parse_document(path)

            self.assertEqual(
                [(link.destination, link.line) for link in document.links],
                [("target.md", 3), ("target.md#section", 4), ("assets/plot%20one.png", 6)],
            )

    def test_status_examples_inside_multiline_inline_code_are_not_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "note.md"
            path.write_text(
                "# Note\n\n> `example\n> 状态：提纲\n> end`\n> 状态：草稿\n"
                "> <!-- hidden\n> 状态：已归档\n> -->\n> 状态：已核对\n",
                encoding="utf-8",
            )

            document = parse_document(path)

            self.assertEqual(document.statuses, [("草稿", 6), ("已核对", 10)])


class NoteLinkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scope = RepositoryScope(self.root)

    def write(self, relative, text=""):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_missing_local_target_is_reported_but_valid_encoded_resources_pass(self):
        self.write("assets/plot one.png", "image placeholder")
        self.write("01_topic/target.md", "# Target\n")
        source = self.write(
            "01_topic/source.md",
            "# Source\n\n[valid][target]\n![plot](../assets/plot%20one.png)\n"
            "[missing](absent.md)\n[remote](https://example.invalid/no-network)\n\n"
            "[target]: target.md\n",
        )

        diagnostics = check_links(self.scope, [source])

        self.assertEqual([(d.line, d.code, d.severity) for d in diagnostics], [(5, "LINK_MISSING", "error")])
        self.assertEqual(diagnostics[0].path, source)

    def test_repository_boundaries_do_not_depend_on_submodule_checkout(self):
        self.write(".gitmodules", '[submodule "nope"]\npath = topics/nope\nurl = https://example.invalid/nope.git\n')
        self.scope = RepositoryScope(self.root)
        source = self.write(
            "01_topic/source.md",
            "# Source\n\n[topic](../topics/nope/README.md#result)\n"
            "[backup](../_backups/secret.md)\n[outside](../../outside.md)\n",
        )

        before = check_links(self.scope, [source])
        module_document = self.write("topics/nope/README.md")
        module_document.write_bytes(b"\xff")
        after = check_links(self.scope, [source])

        expected = [
            (3, "LINK_SUBMODULE", "warning"),
            (4, "LINK_BOUNDARY", "error"),
            (5, "LINK_BOUNDARY", "error"),
        ]
        self.assertEqual([(d.line, d.code, d.severity) for d in before], expected)
        self.assertEqual(after, before)

    def test_path_spelling_is_case_sensitive_on_every_platform(self):
        self.write("assets/Figure.png", "image placeholder")
        self.write("01_topic/Target.md", "# Target\n")
        source = self.write(
            "01_topic/source.md",
            "# Source\n\n[wrong file](target.md)\n"
            "![wrong directory](../Assets/Figure.png)\n[correct](Target.md)\n",
        )

        diagnostics = check_links(self.scope, [source])

        self.assertEqual([(d.line, d.code) for d in diagnostics], [(3, "LINK_CASE"), (4, "LINK_CASE")])

    def test_heading_and_explicit_anchors_support_duplicates_and_encoded_fragments(self):
        self.write(
            "01_topic/target.md",
            '# Target\n\n## 中文 **标题** 与 `code`\n\n## 中文 **标题** 与 `code`\n\n'
            '<a id="stable-result"></a>\n\n<a name="legacy-result"></a>\n',
        )
        source = self.write(
            "01_topic/source.md",
            "# Source\n\n[first](target.md#中文-标题-与-code)\n"
            "[second](target.md#中文-标题-与-code-1)\n"
            "[encoded](target.md#%E4%B8%AD%E6%96%87-%E6%A0%87%E9%A2%98-%E4%B8%8E-code)\n"
            "[explicit](target.md#stable-result)\n[legacy](target.md#legacy-result)\n"
            "[own](#source)\n[missing](target.md#not-there)\n",
        )

        diagnostics = check_links(self.scope, [source])

        self.assertEqual([(d.line, d.code, d.severity) for d in diagnostics], [(9, "LINK_ANCHOR", "error")])

    def test_complex_or_non_markdown_anchors_are_unverified_not_broken(self):
        self.write("01_topic/math.md", '# Math\n\n## $QK^T$\n\n<a id="stable"></a>\n')
        self.write("01_topic/html.md", "# HTML\n\n<h2><em>Custom heading</em></h2>\n")
        self.write("assets/report.pdf", "PDF placeholder")
        source = self.write(
            "01_topic/source.md",
            "# Source\n\n[math](math.md#qkt)\n[html](html.md#custom-heading)\n"
            "[stable](math.md#stable)\n[pdf](../assets/report.pdf#page=2)\n",
        )

        diagnostics = check_links(self.scope, [source])

        self.assertEqual(
            [(d.line, d.code, d.severity) for d in diagnostics],
            [(3, "LINK_ANCHOR_UNVERIFIED", "warning"), (4, "LINK_ANCHOR_UNVERIFIED", "warning"), (6, "LINK_ANCHOR_UNVERIFIED", "warning")],
        )

    def test_public_guides_can_link_to_templates_without_validating_placeholders(self):
        self.write("templates/note.md", "# {{title}}\n\n> 状态：{{status}}\n\n$unfinished\n")
        source = self.write(
            "STYLE_GUIDE.md",
            "# Writing\n\n[template](templates/note.md)\n"
            "[section](templates/note.md#title)\n[backup](_backups/old.md)\n",
        )

        diagnostics = check_links(self.scope, [source])

        self.assertEqual(
            [(d.line, d.code, d.severity) for d in diagnostics],
            [(4, "LINK_ANCHOR_UNVERIFIED", "warning"), (5, "LINK_BOUNDARY", "error")],
        )

    def test_repository_root_links_resolve_from_root_and_keep_case_checks(self):
        self.write("01_topic/target.md", "# Target\n")
        self.write("assets/Figure.png", "image placeholder")
        source = self.write(
            "01_topic/source.md",
            "# Source\n\n[from root](/01_topic/target.md#target)\n"
            "![image](/assets/Figure.png)\n![wrong case](/Assets/Figure.png)\n",
        )

        diagnostics = check_links(self.scope, [source])

        self.assertEqual([(d.line, d.code) for d in diagnostics], [(5, "LINK_CASE")])

    def test_windows_absolute_paths_are_not_mistaken_for_external_web_schemes(self):
        source = self.write("01_topic/source.md", "# Source\n\n[local machine](C:/private/note.md)\n")

        diagnostics = check_links(self.scope, [source])

        self.assertEqual([(d.line, d.code) for d in diagnostics], [(3, "LINK_BOUNDARY")])

    def test_backslashes_in_local_urls_are_rejected_on_every_platform(self):
        self.write("01_topic/target.md", "# Target\n")
        source = self.write(
            "01_topic/source.md",
            "# Source\n\n[encoded](.%5Ctarget.md)\n[slash](./target.md)\n",
        )

        diagnostics = check_links(self.scope, [source])

        self.assertEqual([(d.line, d.code) for d in diagnostics], [(3, "LINK_PATH")])


if __name__ == "__main__":
    unittest.main()
