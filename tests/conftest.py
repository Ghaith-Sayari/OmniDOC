"""
Shared pytest fixtures for the OmniDoc test suite.
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def sample_docs_dir(tmp_path_factory) -> Path:
    """Create a small sample document directory used across integration tests."""
    base = tmp_path_factory.mktemp("sample_docs")

    # Markdown file
    (base / "guide.md").write_text(
        "---\ntitle: OmniDoc Guide\n---\n# Introduction\n\nThis is a test document "
        "for the OmniDoc RAG integration test suite. " * 10,
        encoding="utf-8",
    )

    # CSV file
    (base / "data.csv").write_text(
        "product,revenue,region\nWidgetA,100000,EMEA\nWidgetB,250000,APAC\nWidgetC,175000,AMER\n",
        encoding="utf-8",
    )

    # HTML file
    (base / "page.html").write_text(
        "<html><body><h1>Enterprise Policy</h1><p>All data must be encrypted at rest.</p></body></html>",
        encoding="utf-8",
    )

    return base


@pytest.fixture
def ragas_sample_records():
    """Minimal Q&A records compatible with the Ragas evaluator."""
    return [
        {
            "question": "What is RAG?",
            "answer": "RAG stands for Retrieval-Augmented Generation [chunk:c1].",
            "contexts": ["RAG stands for Retrieval-Augmented Generation, combining retrieval and LLMs."],
            "ground_truth": "RAG is Retrieval-Augmented Generation.",
        }
    ]
