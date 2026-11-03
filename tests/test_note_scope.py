from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from note_tools.scope import RepositoryScope
import check_markdown_math as math_checker


class ScopeTests(unittest.TestCase):
    def test_explicit_math_fix_cannot_enter_a_repository_inside_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.git').mkdir()
            nested = root / '_backups/clone'
            (nested / '.git').mkdir(parents=True)
            note = nested / 'note.md'
            original = '正文，$x$。\n'.encode()
            note.write_bytes(original)

            self.assertEqual(math_checker.find_markdown_files([str(note)]), [])
            self.assertEqual(math_checker.main(['--fix', str(note)]), 2)
            self.assertEqual(note.read_bytes(), original)

    def test_explicit_repository_root_symlink_is_not_scanned_or_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target'
            (target / '.git').mkdir(parents=True)
            note = target / 'note.md'
            original = '正文，$x$。\n'.encode()
            note.write_bytes(original)
            alias = root / 'alias'
            try:
                alias.symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f'symlink creation unavailable: {error}')

            with self.assertRaisesRegex(ValueError, 'symlink|junction'):
                RepositoryScope(alias)
            self.assertEqual(math_checker.main(['--fix', str(alias / 'note.md')]), 2)
            self.assertEqual(note.read_bytes(), original)

    def test_untracked_notes_and_submodule_boundaries_are_independent_of_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.gitmodules').write_text(
                '[submodule "nope"]\n\tpath = projects/nope\n'
                '\turl = https://example.org/nope.git\n', encoding='utf-8')
            for name in ['README.md', '04_transformer/README.md',
                         '04_transformer/new_note.md', '_backups/bad.md',
                         'templates/note.md', '04_transformer/code/README.md']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# Example\n', encoding='utf-8')
            scope = RepositoryScope(root)
            self.assertEqual(scope.note_files(), [root / '04_transformer/new_note.md'])
            before = scope.markdown_files()
            module_file = root / 'projects/nope/bad.md'
            self.assertEqual(scope.boundary(module_file), 'submodule')
            module_file.parent.mkdir(parents=True)
            module_file.write_text('bad', encoding='utf-8')
            self.assertEqual(scope.markdown_files(), before)
            self.assertEqual(scope.boundary(root / '_backups/bad.md'), 'backup')

    def test_existing_math_fix_respects_declared_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.gitmodules').write_text(
                '[submodule "topic"]\npath = projects/topic\nurl = https://example.org/topic.git\n',
                encoding='utf-8')
            source = '正文，$x$。\r\n'
            for name in ['article.md', '_backups/saved.md', 'projects/topic/note.md']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode())
            self.assertEqual(math_checker.main(['--fix', str(root)]), 0)
            self.assertEqual((root / 'article.md').read_bytes(), '正文， $x$。\r\n'.encode())
            for name in ['_backups/saved.md', 'projects/topic/note.md']:
                self.assertEqual((root / name).read_bytes(), source.encode())

    def test_git_config_quotes_and_inline_comments_do_not_leak_submodules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.gitmodules').write_text(
                '[submodule "nope"]\npath = projects/nope # explanation\n'
                '[submodule "quoted"]\npath = "projects/with # name" ; comment\n', encoding='utf-8')
            for name in ['projects/nope/note.md', 'projects/with # name/note.md']:
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_text('# External\n', encoding='utf-8')
            scope = RepositoryScope(root)
            self.assertEqual(scope.markdown_files(), [])
            self.assertEqual(scope.boundary(root / 'projects/nope/note.md'), 'submodule')

    def test_explicit_math_input_cannot_escape_parent_submodule_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.gitmodules').write_text('[submodule "topic"]\npath = projects/topic\n', encoding='utf-8')
            module = root / 'projects/topic'
            module.mkdir(parents=True)
            note = module / 'note.md'
            original = '正文，$x$。\n'.encode()
            note.write_bytes(original)
            for initialized in [False, True]:
                with self.subTest(initialized=initialized):
                    if initialized:
                        (module / '.git').write_text('gitdir: ../../.git/modules/topic\n', encoding='utf-8')
                    self.assertEqual(math_checker.find_markdown_files([str(note)]), [])
                    math_checker.main(['--fix', str(note)])
                    self.assertEqual(note.read_bytes(), original)

    def test_main_scan_excludes_nested_repositories_but_explicit_independent_input_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.git').mkdir()
            main = root / 'main.md'
            main.write_text('# Main\n', encoding='utf-8')
            for marker in ['directory', 'worktree_file']:
                with self.subTest(marker=marker):
                    nested = root / marker
                    nested.mkdir()
                    if marker == 'directory':
                        (nested / '.git').mkdir()
                    else:
                        (nested / '.git').write_text('gitdir: elsewhere\n', encoding='utf-8')
                    note = nested / 'note.md'
                    original = '正文，$x$。\n'.encode()
                    note.write_bytes(original)
                    self.assertEqual(RepositoryScope(root).markdown_files(), [main])
                    self.assertEqual(math_checker.main(['--fix', str(root)]), 0)
                    self.assertEqual(note.read_bytes(), original)
                    self.assertEqual(math_checker.find_markdown_files([str(note)]), [note])


if __name__ == '__main__':
    unittest.main()
