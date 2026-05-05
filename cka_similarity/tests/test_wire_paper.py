"""Tests for wire_paper_results.substitute_todos (Phase-7 Task 7.8)."""


def test_wire_substitutes_todos(tmp_path):
    from wire_paper_results import substitute_todos
    src = tmp_path / "section.tex"
    src.write_text(r"The KM amplification is \TODO{8--16}$\times$ across all attacks.")

    substitute_todos(str(src), {"8--16": "12.4"})
    out = src.read_text()
    assert r"\TODO" not in out
    assert "12.4" in out


def test_wire_preserves_unmatched_todos(tmp_path):
    from wire_paper_results import substitute_todos
    src = tmp_path / "section.tex"
    src.write_text(r"x = \TODO{42} and y = \TODO{99}.")
    substitute_todos(str(src), {"42": "fortytwo"})
    out = src.read_text()
    assert "fortytwo" in out
    assert r"\TODO{99}" in out
