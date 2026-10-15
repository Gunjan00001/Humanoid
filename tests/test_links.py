"""Every relative link in the documentation must point at a file that exists.

A doc that links to a file nobody kept is worse than no link: a reviewer clicks
it and concludes the project is broken. This test walks README.md, DEEPDIVE.md
and every markdown file in docs/, extracts the relative links and images, and
fails listing the ones that do not resolve.

Deliberately not checked: absolute URLs, anchors into other files, and links
inside fenced code blocks.

  python -m unittest tests.test_links
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = [ROOT / "README.md", ROOT / "DEEPDIVE.md"] + sorted((ROOT / "docs").glob("*.md"))

# Markdown inline links and images: [text](target) and ![alt](target)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def relative_links(path: Path) -> list[str]:
    """Link targets in one file that are meant to be files in this repository."""
    text = path.read_text()
    # Drop fenced code blocks so shell examples are not treated as links.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    out = []
    for target in LINK.findall(text):
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:", "#", "<")):
            continue
        target = target.split("#", 1)[0]          # strip the anchor
        if target:
            out.append(target)
    return out


class TestDocumentationLinks(unittest.TestCase):
    def test_doc_list_is_not_empty(self):
        self.assertGreater(len(DOCS), 5, "the documentation list looks wrong")

    def test_relative_links_resolve(self):
        missing: list[str] = []
        for doc in DOCS:
            for target in relative_links(doc):
                resolved = (doc.parent / target).resolve()
                if not resolved.exists():
                    missing.append(f"{doc.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [], "broken relative links:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
