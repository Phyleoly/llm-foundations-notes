from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_markdown_math as checker  # noqa: E402


class MarkdownMathCheckerTests(unittest.TestCase):
    def test_context_aware_fix_is_conservative_and_idempotent(self) -> None:
        source = r"""# Example

Price is $5, inline math is $x_i$, and code is `$$ = $$`.

$$
x
=
\operatorname{id}(y)
$$

```text
$$
ignored
=
\operatorname{id}(ignored)
$$
```

<pre>
$$
also ignored
=
\operatorname{id}(ignored)
$$
</pre>
"""
        analysis = checker.analyze_document(source, "example.md", apply_fixes=True)

        self.assertEqual(
            [issue.code for issue in analysis.issues], ["MATH001", "MATH003"]
        )
        self.assertEqual(analysis.fix_count, 2)
        self.assertIn("```math\nx\n=\n\\mathrm{id}(y)\n```", analysis.text)
        self.assertIn("```text\n$$\nignored\n=", analysis.text)
        self.assertIn("<pre>\n$$\nalso ignored\n=", analysis.text)

        clean = checker.analyze_document(
            analysis.text, "example.md", apply_fixes=True
        )
        self.assertEqual(clean.issues, [])
        self.assertEqual(clean.fix_count, 0)
        self.assertEqual(clean.text, analysis.text)

    def test_unknown_and_starred_operators_are_never_rewritten(self) -> None:
        source = r"""```math
\operatorname{custom}(x)
\operatorname*{softmax}(x)
```
"""
        analysis = checker.analyze_document(source, "operators.md", apply_fixes=True)

        self.assertEqual([issue.code for issue in analysis.issues], ["MATH003", "MATH003"])
        self.assertTrue(all(not issue.fixable for issue in analysis.issues))
        self.assertEqual(analysis.fix_count, 0)
        self.assertEqual(analysis.text, source)

    def test_even_simple_dollar_blocks_use_math_fences(self) -> None:
        source = "$$\n\\mathcal V=\\{v_1,v_2\\}\n$$\n"
        analysis = checker.analyze_document(source, "display.md", apply_fixes=True)

        self.assertEqual([issue.code for issue in analysis.issues], ["MATH001"])
        self.assertEqual(analysis.fix_count, 1)
        self.assertEqual(
            analysis.text,
            "```math\n\\mathcal V=\\{v_1,v_2\\}\n```\n",
        )
        self.assertEqual(
            checker.analyze_document(analysis.text, "display.md").issues, []
        )

    def test_inline_operator_is_fixed_but_inline_code_is_ignored(self) -> None:
        source = (
            r"Use $\operatorname{softmax}(z)$, keep `"
            r"$\operatorname{softmax}(z)$` literal, and pay $20."
            "\n"
        )
        analysis = checker.analyze_document(source, "inline.md", apply_fixes=True)

        self.assertEqual(len(analysis.issues), 1)
        self.assertEqual(analysis.issues[0].code, "MATH003")
        self.assertEqual(analysis.fix_count, 1)
        self.assertIn(r"$\mathrm{softmax}(z)$", analysis.text)
        self.assertIn(r"`$\operatorname{softmax}(z)$`", analysis.text)
        self.assertNotIn("MATH004", [issue.code for issue in analysis.issues])

    def test_html_sensitive_comparisons_are_rewritten_only_in_math(self) -> None:
        source = r"""Inline $x_{<i}$ and $p>0$.

```math
0<\exp(z)&lt;2
```

`$x_{<i}$`

```text
$p>0$
```
"""
        analysis = checker.analyze_document(source, "angles.md", apply_fixes=True)

        self.assertEqual(
            [issue.code for issue in analysis.issues],
            ["MATH007", "MATH007", "MATH007", "MATH007"],
        )
        self.assertEqual(analysis.fix_count, 4)
        self.assertIn(r"$x_{\lt i}$", analysis.text)
        self.assertIn(r"$p\gt 0$", analysis.text)
        self.assertIn(r"0\lt \exp(z)\lt 2", analysis.text)
        self.assertIn(r"`$x_{<i}$`", analysis.text)
        self.assertIn("```text\n$p>0$\n```", analysis.text)

        clean = checker.analyze_document(analysis.text, "angles.md")
        self.assertEqual(clean.issues, [])

    def test_reports_unclosed_delimiters_and_latex_structure(self) -> None:
        inline = checker.analyze_document("Broken $x_i\n", "inline.md")
        display = checker.analyze_document("$$\nx=y\n", "display.md")
        structure = checker.analyze_document(
            "```math\n\\begin{aligned}\n{x\n```\n", "structure.md"
        )

        self.assertIn("MATH004", [issue.code for issue in inline.issues])
        self.assertIn("MATH002", [issue.code for issue in display.issues])
        self.assertEqual(
            {issue.code for issue in structure.issues}, {"MATH005", "MATH006"}
        )

    def test_attention_diag_after_chinese_comma_is_reported_and_fixed(self) -> None:
        source = (
            r"其中，$\mathrm{diag}(\mathbf p)$ 的对角线元素为 $p_1,\dots,p_n$，"
            r"其余元素为 $0$。"
            "\n"
        )
        checked = checker.analyze_document(source, "attention.md")
        self.assertEqual([issue.code for issue in checked.issues], ["MATH008"])
        self.assertEqual(checked.issues[0].line, 1)
        self.assertTrue(checked.issues[0].fixable)
        self.assertEqual(checked.text, source)
        self.assertEqual(checked.fix_count, 0)

        fixed = checker.analyze_document(source, "attention.md", apply_fixes=True)
        self.assertEqual(fixed.text, source.replace("其中，$", "其中， $"))
        self.assertEqual(fixed.fix_count, 1)
        again = checker.analyze_document(fixed.text, apply_fixes=True)
        self.assertEqual(again.issues, [])
        self.assertEqual(again.fix_count, 0)
        self.assertEqual(again.text, fixed.text)

    def test_inline_math_after_prose_punctuation_gets_an_ascii_space(self) -> None:
        for punctuation in "，。；：！？、（）【】《》“”‘’…—,.;:!?":
            with self.subTest(punctuation=punctuation):
                source = f"正文{punctuation}$x$。\n"
                fixed = checker.analyze_document(source, apply_fixes=True)
                self.assertEqual([issue.code for issue in fixed.issues], ["MATH008"])
                self.assertEqual(fixed.text, f"正文{punctuation} $x$。\n")
                self.assertEqual(fixed.fix_count, 1)

    def test_boundary_and_tex_fixes_on_the_same_line_keep_offsets(self) -> None:
        source = r"其中，$\operatorname{softmax}(x)$、$p>0$；$z$。" + "\n"
        fixed = checker.analyze_document(source, apply_fixes=True)
        self.assertEqual(
            fixed.text, r"其中， $\mathrm{softmax}(x)$、 $p\gt 0$； $z$。" + "\n"
        )
        self.assertEqual(fixed.fix_count, 5)
        self.assertEqual(checker.analyze_document(fixed.text).issues, [])

    def test_boundary_fix_ignores_code_html_display_math_and_currency(self) -> None:
        source = r"""Keep `其中，$x$` and <code>其中，$x$</code> literal.
<!-- 其中，$x$ -->
<!--
其中，$x$
-->
<pre>
其中，$x$
</pre>
```text
其中，$x$
```
```math
\text{其中，$x$}
```
价格：$20，转义：\$x\$。
"""
        fixed = checker.analyze_document(source, apply_fixes=True)
        self.assertEqual(fixed.issues, [])
        self.assertEqual(fixed.text, source)
        self.assertEqual(fixed.fix_count, 0)

    def test_supported_inline_boundaries_and_entities_are_preserved(self) -> None:
        source = r"""$x$，其中 $y$，其中， $z$，以及 ($w$)。
**$x$** and [$y$](example.md) and 其中，$`\mathrm{diag}(\mathbf p)`$。
$\mathrm{diag}(\mathbf p)$&#x20;
其中，$\mathrm{diag}(\mathbf p)$&#x20;
"""
        fixed = checker.analyze_document(source, apply_fixes=True)
        self.assertEqual([issue.code for issue in fixed.issues], ["MATH008"])
        self.assertEqual(fixed.text, source.replace("其中，$\\mathrm", "其中， $\\mathrm"))
        self.assertEqual(fixed.text.count("&#x20;"), 2)

    def test_boundary_file_fix_preserves_crlf_and_missing_final_newline(self) -> None:
        source = "# CRLF\r\n\r\n其中，$x$、$y$。"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.md"
            path.write_bytes(source.encode("utf-8"))
            issues, changed, fix_count = checker.process_file(path, fix=True)
            self.assertEqual(issues, [])
            self.assertTrue(changed)
            self.assertEqual(fix_count, 2)
            self.assertEqual(
                path.read_bytes(), "# CRLF\r\n\r\n其中， $x$、 $y$。".encode("utf-8")
            )
            self.assertEqual(checker.process_file(path, fix=True), ([], False, 0))

    def test_file_fix_preserves_crlf_and_is_idempotent(self) -> None:
        source = (
            "# CRLF\r\n\r\n$$\r\nx\r\n=\r\n"
            "\\operatorname{Lik}(x)\r\n$$\r\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.md"
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(source)

            issues, changed, fix_count = checker.process_file(path, fix=True)
            with path.open("r", encoding="utf-8", newline="") as handle:
                fixed = handle.read()

            self.assertEqual(issues, [])
            self.assertTrue(changed)
            self.assertEqual(fix_count, 2)
            self.assertNotIn("\n", fixed.replace("\r\n", ""))
            self.assertIn("```math\r\n", fixed)
            self.assertIn(r"\mathrm{Lik}(x)", fixed)

            issues, changed, fix_count = checker.process_file(path, fix=True)
            self.assertEqual(issues, [])
            self.assertFalse(changed)
            self.assertEqual(fix_count, 0)


if __name__ == "__main__":
    unittest.main()
