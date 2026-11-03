from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from markdown_it import MarkdownIt


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / 'scripts/notes.py'


class AuthoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / '04_transformer').mkdir()
        (self.root / 'templates').mkdir()
        (self.root / 'templates/note.md').write_bytes((REPO / 'templates/note.md').read_bytes())

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(CLI), '--root', str(self.root), *args],
                              capture_output=True, encoding='utf-8',
                              env=dict(os.environ, PYTHONUTF8='1'))

    def test_create_plain_markdown_without_renaming_or_overwriting(self):
        old = self.root / '04_transformer/01_attention.md'
        old.write_bytes(b'# Existing\r\n')
        result = self.run_cli('new', '04_transformer/grouped_query_attention.md',
                              '--title', 'Grouped-Query Attention')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        new = self.root / '04_transformer/grouped_query_attention.md'
        content = new.read_bytes()
        self.assertIn('> 状态：草稿'.encode(), content)
        self.assertEqual(old.read_bytes(), b'# Existing\r\n')
        result = self.run_cli('new', '04_transformer/grouped_query_attention.md', '--title', 'Other')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(new.read_bytes(), content)
        result = self.run_cli('new', '../outside.md', '--title', 'Outside')
        self.assertNotEqual(result.returncode, 0)

    def test_generated_inventory_preserves_reading_route_and_newlines(self):
        readme = self.root / '04_transformer/README.md'
        prefix = '# Chapter\r\n\r\nRead B before A.\r\n\r\n<!-- notes:index:start -->'
        suffix = '<!-- notes:index:end -->\r\n\r\nKeep this ending.'
        readme.write_bytes((prefix + '\r\n' + suffix).encode())
        note = self.root / '04_transformer/new_note.md'
        note.write_text('# 中文标题\n\n> 状态：草稿\n\n本篇要回答：什么？\n', encoding='utf-8')
        stale = self.run_cli('index', '--check')
        self.assertEqual(stale.returncode, 1, stale.stdout + stale.stderr)
        result = self.run_cli('index')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        first = readme.read_bytes()
        self.assertTrue(first.startswith(prefix.encode()))
        self.assertTrue(first.endswith(suffix.encode()))
        self.assertIn('[中文标题](./new_note.md) | 草稿'.encode(), first)
        self.assertNotIn(b'\n', first.replace(b'\r\n', b''))
        self.assertEqual(self.run_cli('index').returncode, 0)
        self.assertEqual(self.run_cli('index', '--check').returncode, 0)
        self.assertEqual(readme.read_bytes(), first)
        note.write_text('# Changed\n\n> 状态：提纲\n', encoding='utf-8')
        self.assertEqual(self.run_cli('index', '--check').returncode, 1)
        self.assertEqual(readme.read_bytes(), first)

    def test_invalid_template_is_rejected_before_creating_files_or_directories(self):
        for template in [
            '## {{ title }}\n\n> 状态：草稿\n',
            '# {{ title }}\n\n> 状态：未知\n',
            '# {{ title }}\n\n> 状态：已核对\n',
        ]:
            with self.subTest(template=template):
                (self.root / 'templates/note.md').write_text(template, encoding='utf-8')
                result = self.run_cli('new', '04_transformer/subtopic/draft.md', '--title', 'Draft')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / '04_transformer/subtopic').exists())

    def test_production_templates_support_new_index_check_workflow(self):
        chapter = (REPO / 'templates/chapter_readme.md').read_text(encoding='utf-8')
        (self.root / '04_transformer/README.md').write_text(chapter.replace('{{ title }}', 'Chapter'), encoding='utf-8')
        for name in ['note', 'paper_reading']:
            (self.root / f'templates/{name}.md').write_bytes((REPO / f'templates/{name}.md').read_bytes())
            result = self.run_cli('new', f'04_transformer/{name}.md', '--title', name, '--template', name)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for arguments in [('index',), ('check',), ('index', '--check')]:
            result = self.run_cli(*arguments)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_new_can_select_a_paper_template_without_overwriting(self):
        (self.root / 'templates/paper_reading.md').write_text(
            '# {{ title }}\n\n> 状态：草稿\n\n本篇要回答：\n\n论文来源：\n', encoding='utf-8')
        args = ('new', '04_transformer/paper_notes.md', '--title', '论文阅读', '--template', 'paper_reading')
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        target = self.root / '04_transformer/paper_notes.md'
        before = target.read_bytes()
        self.assertIn('论文来源：'.encode(), before)
        self.assertNotEqual(self.run_cli(*args).returncode, 0)
        self.assertEqual(target.read_bytes(), before)

    def test_new_rejects_non_article_templates(self):
        for template in ['chapter_readme', '../outside']:
            with self.subTest(template=template):
                result = self.run_cli('new', '04_transformer/new_note.md', '--title', 'Title', '--template', template)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / '04_transformer/new_note.md').exists())

    def test_check_is_read_only_and_catches_a_note_missing_from_clean_checkout(self):
        readme = self.root / '04_transformer/README.md'
        readme.write_text('# Chapter\n\n<!-- notes:index:start -->\n<!-- notes:index:end -->\n', encoding='utf-8')
        note = self.root / '04_transformer/draft.md'
        note.write_text('# Draft\n\n> 状态：草稿\n\n本篇要回答：示例问题。\n', encoding='utf-8')
        self.assertEqual(self.run_cli('index').returncode, 0)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = self.run_cli('check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        note.unlink()  # Simulate committing the inventory without the new article.
        result = self.run_cli('check')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('draft.md', result.stdout)

    def test_generated_titles_preserve_literal_code_and_markdown_characters(self):
        readme = self.root / '04_transformer/README.md'
        readme.write_text('# Chapter\n\n<!-- notes:index:start -->\n<!-- notes:index:end -->\n', encoding='utf-8')
        (self.root / '04_transformer/literal_title.md').write_text(
            '# `price$usd` and `<token>` & `[x]|*_~`\n\n> 状态：草稿\n', encoding='utf-8')

        result = self.run_cli('index')

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rendered = MarkdownIt('commonmark').enable('table').render(readme.read_text(encoding='utf-8'))
        self.assertIn('<a href="./literal_title.md">price$usd and &lt;token&gt; &amp; [x]|*_~</a>', rendered)
        result = self.run_cli('check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_marker_in_later_chapter_does_not_partially_write(self):
        first = self.root / '04_transformer/README.md'
        first.write_bytes(b'# Chapter\r\n<!-- notes:index:start -->\r\n<!-- notes:index:end -->')
        second = self.root / '05_training'
        second.mkdir()
        (second / 'README.md').write_text('# Missing markers\n', encoding='utf-8')
        before = first.read_bytes()
        self.assertNotEqual(self.run_cli('index').returncode, 0)
        self.assertEqual(first.read_bytes(), before)

    def test_marker_examples_in_a_fence_are_never_replaced(self):
        readme = self.root / '04_transformer/README.md'
        source = '# Chapter\n\n```markdown\n<!-- notes:index:start -->\nExample prose\n<!-- notes:index:end -->\n```\n'
        readme.write_text(source, encoding='utf-8')
        result = self.run_cli('index')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(readme.read_text(encoding='utf-8'), source)

    def test_chapter_readme_case_is_portable(self):
        readme = self.root / '04_transformer/readme.md'
        original = b'# Chapter\n<!-- notes:index:start -->\n<!-- notes:index:end -->\n'
        readme.write_bytes(original)
        self.assertNotEqual(self.run_cli('index').returncode, 0)
        self.assertEqual(readme.read_bytes(), original)

    def test_symlinked_chapter_readme_is_not_modified(self):
        target = self.root / 'outside.md'
        source = b'# Original\n<!-- notes:index:start -->\n<!-- notes:index:end -->\n'
        target.write_bytes(source)
        readme = self.root / '04_transformer/README.md'
        try:
            readme.symlink_to(target)
        except OSError as error:
            self.skipTest(f'symlink creation unavailable: {error}')
        self.assertNotEqual(self.run_cli('index').returncode, 0)
        self.assertEqual(target.read_bytes(), source)


if __name__ == '__main__':
    unittest.main()
