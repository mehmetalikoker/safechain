from __future__ import annotations
from typing import List, Optional

from safechain.schema import Document


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, seps: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = next((s for s in seps if s and s in text), "")
        if not sep:
            # Hard cut with overlap
            chunks: List[str] = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                chunk = text[i : i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        parts = text.split(sep)
        remaining_seps = seps[seps.index(sep) + 1 :]
        result: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    if len(current) > self.chunk_size and remaining_seps:
                        result.extend(self._split(current, remaining_seps))
                    else:
                        result.append(current)
                current = part

        if current.strip():
            if len(current) > self.chunk_size and remaining_seps:
                result.extend(self._split(current, remaining_seps))
            else:
                result.append(current)

        return result

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result: List[Document] = []
        for doc in documents:
            for i, chunk in enumerate(self.split_text(doc.page_content)):
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i},
                ))
        return result


class CharacterTextSplitter:
    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        parts = text.split(self.separator)
        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = (current + self.separator + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result: List[Document] = []
        for doc in documents:
            for i, chunk in enumerate(self.split_text(doc.page_content)):
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i},
                ))
        return result
