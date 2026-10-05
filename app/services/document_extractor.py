from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO

from docx import Document
from pypdf import PdfReader


@dataclass
class ExtractedContent:
    """Structured content extracted from a document."""

    text: str
    metadata: dict


class DocumentExtractor(ABC):
    """Base interface for extracting text from documents."""

    @abstractmethod
    def extract(self, content: bytes) -> str:
        """Extract plain text from raw document bytes."""
        raise NotImplementedError

    def extract_structured(self, content: bytes) -> list[ExtractedContent]:
        """
        Extract document content together with metadata.

        The default implementation preserves the existing extractor
        behavior and returns a single content block.
        """
        return [
            ExtractedContent(
                text=self.extract(content),
                metadata={},
            )
        ]


class TextDocumentExtractor(DocumentExtractor):
    """Extract text from UTF-8 plain-text documents."""

    def extract(self, content: bytes) -> str:
        return content.decode("utf-8")

    def extract_structured(
        self,
        content: bytes,
    ) -> list[ExtractedContent]:
        return [
            ExtractedContent(
                text=self.extract(content),
                metadata={},
            )
        ]


class PDFDocumentExtractor(DocumentExtractor):
    """Extract text from PDF documents while preserving page metadata."""

    def extract(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n\n".join(pages).strip()

    def extract_structured(
        self,
        content: bytes,
    ) -> list[ExtractedContent]:
        reader = PdfReader(BytesIO(content))

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                extracted_pages.append(
                    ExtractedContent(
                        text=text.strip(),
                        metadata={
                            "page": page_number,
                        },
                    )
                )

        return extracted_pages


class DOCXDocumentExtractor(DocumentExtractor):
    """Extract text from DOCX documents."""

    def extract(self, content: bytes) -> str:
        document = Document(BytesIO(content))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n\n".join(paragraphs).strip()

    def extract_structured(
        self,
        content: bytes,
    ) -> list[ExtractedContent]:
        return [
            ExtractedContent(
                text=self.extract(content),
                metadata={},
            )
        ]