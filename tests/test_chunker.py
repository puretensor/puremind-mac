import itertools
import random
import re
from pathlib import Path

from puremind_mac import chunker
from puremind_mac.chunker import (
    MAX_HEADING_CHARS,
    chunk_markdown,
    clip_heading,
    fenced_blocks,
    fenced_lines,
    frontmatter_end,
)


def test_empty_and_whitespace():
    assert chunk_markdown("") == []
    assert chunk_markdown("  \n\n ") == []


def test_plain_text_is_one_chunk_without_heading():
    (c,) = chunk_markdown("Just a paragraph of text.")
    assert c.heading_path == "" and c.index == 0


def test_splits_on_headings_with_breadcrumb():
    body = "x" * 120
    text = f"# A\n{body}\n\n## B\n{body}\n\n### C\n{body}\n\n## D\n{body}\n"
    paths = [c.heading_path for c in chunk_markdown(text)]
    assert paths == ["A", "A > B", "A > B > C", "A > D"]


def test_heading_levels_above_four_do_not_split():
    text = "# A\n" + "y" * 120 + "\n\n##### Deep\n" + "z" * 120
    assert len(chunk_markdown(text)) == 1


def test_chunk_indexes_are_sequential():
    text = "\n\n".join(f"# H{i}\n" + "w" * 150 for i in range(5))
    assert [c.index for c in chunk_markdown(text)] == [0, 1, 2, 3, 4]


def test_tiny_sections_merge_into_previous_sibling():
    text = "# A\n" + "q" * 150 + "\n\n## B\nshort\n\n## C\nalso short\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 2
    assert "short" in chunks[1].content and "also short" in chunks[1].content


def test_heading_inside_backtick_fence_is_not_a_split():
    text = "# Real\n" + "a" * 120 + "\n\n```bash\n# not a heading\necho hi\n```\n\nafter\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert "# not a heading" in chunks[0].content


def test_heading_inside_tilde_fence_is_not_a_split():
    text = "# Real\n" + "a" * 120 + "\n\n~~~\n## fake\n~~~\n"
    assert len(chunk_markdown(text)) == 1


def test_longer_closing_fence_and_nested_shorter_fence():
    lines = ["````md\n", "```\n", "# inner\n", "```\n", "````\n", "# After\n"]
    inside = fenced_lines(lines)
    assert inside == {0, 1, 2, 3, 4}


def test_unclosed_fence_is_plain_text_so_later_headings_still_split():
    text = "# A\n" + "a" * 120 + "\n\n```python\nprint(1)\n\n# B\n" + "b" * 150
    paths = [c.heading_path for c in chunk_markdown(text)]
    assert paths == ["A", "B"]


def test_fence_with_backtick_in_info_string_is_not_a_fence():
    assert 1 not in fenced_lines(["``` `x`\n", "# h\n"])


def test_many_unmatched_fences_stay_fast():
    text = "```py\n" * 20000
    assert chunk_markdown(text)  # would be quadratic if each opener scanned to the end


def test_short_frontmatter_travels_with_the_first_section():
    text = "---\ntitle: Notes\n# not a heading\ntags: [a]\n---\n\n# Title\n" + "t" * 150
    (c,) = chunk_markdown(text)
    assert c.content.startswith("---\ntitle: Notes")
    assert c.heading_path == "Title"


def test_long_frontmatter_stays_intact_and_headings_inside_it_do_not_split():
    fm = "---\n" + "".join(f"key{i}: value {i}\n" for i in range(30)) + "# not a heading\n---\n"
    chunks = chunk_markdown(fm + "\n# Title\n" + "t" * 150)
    assert chunks[0].heading_path == "" and "# not a heading" in chunks[0].content
    assert chunks[1].heading_path == "Title"


def test_frontmatter_detection():
    assert frontmatter_end(["---\n", "a: 1\n", "---\n"]) == 2
    assert frontmatter_end(["---\n", "a: 1\n"]) is None
    assert frontmatter_end(["text\n", "---\n", "---\n"]) is None


def test_dashes_later_in_file_are_not_frontmatter():
    text = "# A\n" + "p" * 120 + "\n\n---\n\n## B\n" + "q" * 150
    assert [c.heading_path for c in chunk_markdown(text)] == ["A", "A > B"]


def test_oversized_section_is_windowed_to_max_chars():
    text = "# Big\n" + "\n".join(f"line {i} " + "x" * 60 for i in range(200))
    chunks = chunk_markdown(text)
    assert len(chunks) > 1
    assert all(len(c.content) <= 2048 for c in chunks)
    assert {c.heading_path for c in chunks} == {"Big"}


def test_windows_overlap_by_about_twenty_percent():
    text = "".join(f"w{i:05d} " for i in range(1200))  # one long line, no newlines to snap to
    chunks = chunk_markdown(text)
    assert len(chunks) >= 3
    for a, b in itertools.pairwise(chunks):
        assert b.content[:25] in a.content[-450:]  # next window starts inside the last ~20%
    assert len(chunks[0].content) <= 2048


def test_windows_cover_all_text():
    words = [f"tok{i}" for i in range(3000)]
    chunks = chunk_markdown("# T\n" + " ".join(words))
    joined = " ".join(c.content for c in chunks)
    assert all(w in joined for w in words[::97])
    assert "tok2999" in chunks[-1].content


def test_windows_prefer_line_boundaries():
    text = "\n".join("row " + "r" * 70 for _ in range(100))
    chunks = chunk_markdown(text)
    assert len(chunks) > 1
    assert all(c.content.endswith("r") for c in chunks)  # no row cut mid-line at the end


def test_custom_limits():
    chunks = chunk_markdown("a " * 200, max_chars=100)
    assert len(chunks) > 3 and all(len(c.content) <= 100 for c in chunks)


def _fences(content):
    return [line for line in content.splitlines() if line.startswith("```")]


# ---- windows keep fenced blocks and frontmatter whole


def test_a_fenced_block_that_fits_a_window_is_never_cut():
    code = "\n".join(f"call_{i}(argument_{i})  # keep me whole" for i in range(25))  # about 800 characters
    for before in (900, 1100, 1300, 1500, 1700, 1900):
        text = "# Doc\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(before // 220)) + f"\n\n```python\n{code}\n```\n\n" + "tail " * 400
        for chunk in chunk_markdown(text):
            assert len(_fences(chunk.content)) % 2 == 0, (before, chunk.content[:60])
            assert ("call_0(" in chunk.content) == ("call_24(" in chunk.content)


def test_a_fenced_block_longer_than_a_window_is_cut_into_readable_parts():
    code = "\n".join(f"x{i} = compute({i})  # step {i}" for i in range(280))  # about 9 KB
    chunks = chunk_markdown(f"# Fenced\n\nintro\n\n```python\n{code}\n```\n\noutro text\n")
    assert len(chunks) > 3 and all(len(c.content) <= 2048 for c in chunks)
    for chunk in chunks:
        assert _fences(chunk.content)[:1] == ["```python"] or "```python" in chunk.content
        assert len(_fences(chunk.content)) % 2 == 0  # every part opens and closes its own fence
    middle = chunks[1:-1]
    assert middle and all(c.content.startswith("```python\n") and c.content.endswith("\n```") for c in middle)
    joined = "\n".join(c.content for c in chunks)
    assert all(f"x{i} = compute({i})" in joined for i in range(0, 280, 7)) and "outro text" in chunks[-1].content


def test_a_tilde_fence_with_a_long_opener_is_repeated_exactly():
    code = "\n".join(f"line {i} of the block" for i in range(300))
    chunks = chunk_markdown(f"# T\n\n~~~~text title=\"demo\"\n{code}\n~~~~\n")
    assert len(chunks) > 2
    for chunk in chunks:
        assert '~~~~text title="demo"' in chunk.content and chunk.content.rstrip().endswith("~~~~")


def test_a_block_whose_opening_line_is_too_long_to_repeat_is_cut_plainly():
    code = "\n".join(f"line {i} of the block" for i in range(300))
    chunks = chunk_markdown(f"# T\n\n```{'x' * 900}\n{code}\n```\n")
    assert len(chunks) > 2 and all(len(c.content) <= 2048 for c in chunks)
    assert sum(c.content.count("```") for c in chunks) == 2  # nothing is repeated or invented
    assert "line 299 of the block" in chunks[-1].content


def test_frontmatter_longer_than_a_window_is_cut_into_whole_blocks():
    fm = "---\n" + "\n".join(f"key{i}: value number {i} for the frontmatter test" for i in range(85)) + "\n---\n"
    chunks = chunk_markdown(fm + "\n# Heading\n" + "t" * 150)
    head = [c for c in chunks if c.heading_path == ""]
    assert len(head) >= 2 and all(len(c.content) <= 2048 for c in chunks)
    assert all(c.content.startswith("---\n") and c.content.endswith("\n---") for c in head)


def test_windows_end_at_blank_lines_when_there_are_some():
    text = "# Prose\n\n" + "\n\n".join(f"Paragraph {i}. " + "lorem ipsum " * 30 + "end." for i in range(40))
    chunks = chunk_markdown(text)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.content.endswith("end.") and re.match(r"(# Prose|Paragraph \d+)", chunk.content)


# ---- balanced windows that start at word starts


def test_a_section_just_over_the_limit_gives_two_even_windows():
    text = "".join(f"w{i:05d} " for i in range(293))[:2049]  # 2,049 characters on one line
    a, b = chunk_markdown(text)
    assert abs(len(a.content) - len(b.content)) <= 12 and min(len(a.content), len(b.content)) > 1000


def test_windows_start_and_end_at_word_boundaries():
    chunks = chunk_markdown("".join(f"w{i:05d} " for i in range(1200)))
    assert len(chunks) >= 3
    for chunk in chunks:
        assert re.fullmatch(r"(w\d{5} )*w\d{5}", chunk.content), chunk.content[:20]


def test_no_window_is_mostly_a_repeat_of_the_previous_one():
    for length in range(2049, 6000, 97):
        text = "".join(f"w{i:05d} " for i in range(length // 7 + 1))[:length]
        chunks = chunk_markdown(text)
        tail = chunks[-1].content
        new = len(tail) - max((k for k in range(len(tail), 0, -1) if chunks[-2].content.endswith(tail[:k])), default=0)
        assert new > 0.3 * len(tail), (length, len(tail), new)


# ---- bounded breadcrumbs and chunk counts


def test_a_very_long_heading_is_clipped_in_the_breadcrumb():
    chunks = chunk_markdown("# " + "x" * 200_000)
    assert len(chunks) > 1
    assert {len(c.heading_path) for c in chunks} == {MAX_HEADING_CHARS}
    assert chunks[0].content.startswith("# xxxx")  # the text itself is not clipped
    assert clip_heading("short") == "short" and clip_heading("y" * 300) == "y" * 197 + "..."


def test_nested_one_word_headings_do_not_become_one_chunk_each():
    text = "# a\n## b\n### c\n#### d\n" * 5000  # 100 KB
    chunks = chunk_markdown(text)
    assert len(chunks) <= len(text) // 1000  # was one chunk per heading ladder, about 19,000 here


def test_tiny_sections_with_different_parents_pack_up_to_the_minimum_size():
    text = "# a\nx\n## b\ny\n" * 3000
    chunks = chunk_markdown(text)
    assert all(len(c.content) >= 100 for c in chunks[:-1]) and len(chunks) < len(text) // 80


def test_normal_notes_keep_their_chunks():
    body = "x" * 150
    text = f"# Title\n{body}\n\n## One\n{body}\n\n### Deep\n{body}\n\n## Two\n{body}\n\n## Short\nok\n"
    chunks = chunk_markdown(text)
    assert [c.heading_path for c in chunks] == ["Title", "Title > One", "Title > One > Deep", "Title > Two"]
    assert chunks[-1].content.endswith("## Short\nok")  # a tiny section still joins its previous sibling


# ---- a heading without a body travels with its child


def test_a_heading_with_no_body_joins_the_child_section_after_it():
    (chunk,) = chunk_markdown("# Project\n\n## Setup\n\n" + "s" * 233)
    assert chunk.heading_path == "Project > Setup"
    assert chunk.content.startswith("# Project\n\n## Setup\n\nsss")


def test_a_chain_of_bodyless_headings_folds_into_the_first_section_with_text():
    chunks = chunk_markdown("# A\n\n## B\n\n### C\n" + "c" * 120 + "\n\n## D\n" + "d" * 120)
    assert [c.heading_path for c in chunks] == ["A > B > C", "A > D"]
    assert chunks[0].content.startswith("# A\n\n## B\n\n### C\n")


def test_frontmatter_and_a_bodyless_title_travel_together():
    (chunk,) = chunk_markdown("---\ntags: [a]\n---\n\n# Project\n\n## Setup\n\n" + "s" * 200)
    assert chunk.heading_path == "Project > Setup" and chunk.content.startswith("---\ntags: [a]\n---\n\n# Project")


# ---- input quirks


def test_a_leading_bom_does_not_hide_the_first_heading():
    chunks = chunk_markdown("\ufeff# Project\n" + "p" * 120 + "\n\n## Setup\n" + "s" * 120)
    assert [c.heading_path for c in chunks] == ["Project", "Project > Setup"]
    assert frontmatter_end(["\ufeff---\n", "a: 1\n", "---\n"]) == 2


def test_the_source_has_no_invisible_characters():
    source = Path(chunker.__file__).read_text(encoding="utf-8")
    assert "\ufeff" not in source and "\u2014" not in source


def test_fenced_blocks_lists_each_block_with_its_fence_lines():
    lines = ["a\n", "```py\n", "x\n", "```\n", "~~~\n", "y\n", "~~~\n", "```\n", "open\n"]
    assert fenced_blocks(lines) == [(1, 3), (4, 6)]
    assert fenced_lines(lines) == {1, 2, 3, 4, 5, 6}


def _random_document(rng):
    parts = []
    for _ in range(rng.randint(1, 12)):
        kind = rng.choice("hpfbtlmi")
        if kind == "h":
            parts.append("#" * rng.randint(1, 5) + " " + " ".join(rng.choice(["alpha", "beta", "setup"]) for _ in range(3)))
        elif kind == "p":
            parts.append(" ".join(f"w{rng.randint(0, 99999)}" for _ in range(rng.randint(5, 500))))
        elif kind == "f":
            parts.append("```py\n" + "\n".join(f"v{i} = {rng.randint(0, 999)}" for i in range(rng.randint(1, 250))) + "\n```")
        elif kind == "b":
            parts.append("~~~\n" + "\n".join(f"# not a heading {i}" for i in range(rng.randint(1, 40))) + "\n~~~")
        elif kind == "t":
            parts.append("tiny")
        elif kind == "l":
            parts.append("\n".join(f"- item {i} {'z' * rng.randint(0, 60)}" for i in range(rng.randint(1, 90))))
        elif kind == "m":
            parts.append("---\n" + "\n".join(f"k{i}: v{i}" for i in range(rng.randint(1, 150))) + "\n---")
        else:
            parts.append("\n" * rng.randint(0, 3))
    return rng.choice(["\n", "\n\n", "\r\n"]).join(parts)


def test_random_documents_keep_every_chunk_within_the_limit_and_lose_no_word():
    for seed in range(120):
        rng = random.Random(seed)
        text = _random_document(rng)
        limit, small = rng.choice([(2048, 100), (500, 50), (256, 20)])
        chunks = chunk_markdown(text, max_chars=limit, min_chars=small)
        assert all(c.content.strip() and len(c.content) <= limit for c in chunks), seed
        assert chunks == chunk_markdown(text, max_chars=limit, min_chars=small), seed
        pool = "\n".join(c.content for c in chunks)
        lost = [w for w in set(re.findall(r"\S+", text)) if w not in pool and len(w) <= 12 and not w.startswith(("```", "~~~"))]
        assert not lost, (seed, lost[:3])
