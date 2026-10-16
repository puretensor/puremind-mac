from puremind_mac.chunker import chunk_markdown, fenced_lines, frontmatter_end


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
    for a, b in zip(chunks, chunks[1:]):
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
