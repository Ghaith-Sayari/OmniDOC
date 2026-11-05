"""Unit tests for CitationVerifier."""

import pytest
from omnidoc.generation.citation_verifier import CitationVerifier


class TestCitationVerifier:
    def setup_method(self):
        self.verifier = CitationVerifier(strict=False)
        self.chunks = [
            {"id": "abc123", "text": "The sky is blue."},
            {"id": "def456", "text": "Water is H2O."},
        ]

    def test_valid_citations(self):
        answer = "The sky is blue [chunk:abc123]. Water is H2O [chunk:def456]."
        report = self.verifier.verify(answer, self.chunks)
        assert report["valid"] is True
        assert len(report["fabricated_ids"]) == 0

    def test_fabricated_citation_detected(self):
        answer = "Something [chunk:FAKEID]."
        report = self.verifier.verify(answer, self.chunks)
        assert report["valid"] is False
        assert "FAKEID" in report["fabricated_ids"]

    def test_strict_mode_raises(self):
        strict_verifier = CitationVerifier(strict=True)
        answer = "Fabricated [chunk:FAKEID]."
        with pytest.raises(ValueError, match="Fabricated citations"):
            strict_verifier.verify(answer, self.chunks)

    def test_uncited_sentences_flagged(self):
        answer = "This sentence has no citation. This one does [chunk:abc123]."
        report = self.verifier.verify(answer, self.chunks)
        assert len(report["uncited_sentences"]) >= 1

    def test_empty_answer(self):
        report = self.verifier.verify("", self.chunks)
        assert report["valid"] is True
        assert report["cited_ids"] == []
