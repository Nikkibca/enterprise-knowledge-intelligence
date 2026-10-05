import pytest

from app.services.chunking_service import TextChunker


def test_text_is_split_into_chunks():
    chunker = TextChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert chunks[0] == "abcdefghij"


def test_chunks_have_overlap():
    chunker = TextChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunker.split_text(text)

    assert chunks[0][-2:] == chunks[1][:2]


def test_empty_text_returns_empty_list():
    chunker = TextChunker()

    assert chunker.split_text("") == []
    assert chunker.split_text("   ") == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=10,
            chunk_overlap=10,
        )