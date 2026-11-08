"""Synthetic HTML quoting regression; no network or real report generation."""
import unittest

from src.tools.r01_review import quote_in_html


class R01QuoteTests(unittest.TestCase):
    def test_inline_span_does_not_insert_space_before_punctuation(self):
        self.assertTrue(quote_in_html("We introduce MODEL, a framework.",
                                     b'<p>We introduce <span>MODEL</span>, a framework.</p>'))

    def test_visible_word_difference_is_not_accepted(self):
        self.assertFalse(quote_in_html("We introduce MODEL, a framework.",
                                      b'<p>We introduce <span>MODEL</span>, another framework.</p>'))

    def test_whitespace_and_html_entities_preserve_quote(self):
        self.assertTrue(quote_in_html("The agent's model works.",
                                     b'<p>The agent&#8217;s\n model works.</p>'))


if __name__ == "__main__":
    unittest.main()
