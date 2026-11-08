"""Synthetic parser checks only; no network, real bibliography, or research results."""
import json
import unittest

from src.tools.r01_prepare import bibtex, parse_source, protected_parser


class R01SourceParsingTests(unittest.TestCase):
    def setUp(self):
        self.row = {"key": "fixed2026", "title": "A Test: VLM & RL", "arxiv": "2601.00001"}

    def test_official_bibtex_rekeys_and_preserves_conference_metadata(self):
        source = b'@inproceedings{original, title={A Test: VLM & RL}, author={Doe, Jane and Roe, John}, year={2026}, booktitle={Test Conference}, pages={12--19}}'
        entry = parse_source("bibtex", source, self.row)
        self.assertEqual(entry["key"], "fixed2026")
        self.assertEqual(entry["fields"]["booktitle"], "Test Conference")
        self.assertEqual(protected_parser().parse_bib(bibtex(entry))[0]["fields"]["pages"], "12--19")

    def test_html_bibtex_entities(self):
        source = b'<pre>@inproceedings{old, title={A Test: VLM &amp; RL}, author={Jane Doe}, year={2026}}</pre>'
        self.assertEqual(parse_source("html_bibtex", source, self.row)["key"], "fixed2026")

    def test_wrong_title_is_not_silently_substituted(self):
        source = b'@article{old, title={Unrelated paper}, author={Jane Doe}, year={2026}}'
        with self.assertRaises(ValueError):
            parse_source("bibtex", source, self.row)

    def test_incomplete_author_metadata_rejected(self):
        source = b'@article{old, title={A Test: VLM & RL}, year={2026}}'
        with self.assertRaises(ValueError):
            parse_source("bibtex", source, self.row)

    def test_arxiv_atom_rejects_wrong_identifier(self):
        source = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2601.99999</id><title>A Test: VLM &amp; RL</title><published>2026-01-01</published><author><name>Jane Doe</name></author></entry></feed>'
        with self.assertRaises(ValueError):
            parse_source("arxiv_atom", source, self.row)

    def test_arxiv_html_metadata_not_visible_page_authors(self):
        source = b'<meta name="citation_arxiv_id" content="2601.00001v2"><meta name="citation_title" content="A Test: VLM &amp; RL"><meta name="citation_author" content="Jane Doe"><meta name="citation_author" content="John Roe"><meta name="citation_date" content="2026/01/01">'
        entry = parse_source("arxiv_html", source, self.row)
        self.assertEqual(entry["fields"]["author"], "Jane Doe and John Roe")
        self.assertEqual(entry["fields"]["eprint"], "2601.00001")

    def test_crossref_published_year_and_authors(self):
        source = {"message": {"title": ["A Test: VLM & RL"], "author": [{"given": "Jane", "family": "Doe"}],
                              "issued": {"date-parts": [[2026, 1]]}, "DOI": "10.0000/test", "URL": "https://doi.org/10.0000/test",
                              "type": "proceedings-article", "container-title": ["Test Proceedings"], "page": "1-9"}}
        entry = parse_source("crossref", json.dumps(source).encode(), self.row)
        self.assertEqual(entry["type"], "inproceedings")
        self.assertEqual(entry["fields"]["year"], "2026")
        self.assertEqual(entry["fields"]["booktitle"], "Test Proceedings")


if __name__ == "__main__":
    unittest.main()
