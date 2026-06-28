from __future__ import annotations
from typing import List, Optional

from safechain.schema import Document


class RecursiveCharacterTextSplitter:
    """Metni önce büyük ayraçlarda, gerekirse daha küçük ayraçlarda böler.

    Ayraç listesini öncelik sırasıyla dener (çift yeni satır, tek yeni
    satır, nokta, boşluk, karakter). Parça hâlâ büyükse alt ayraçlarla
    yinelemeli olarak devam eder.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ) -> None:
        """RecursiveCharacterTextSplitter oluşturur.

        Args:
            chunk_size: Bir parçanın maksimum karakter uzunluğu. Varsayılan: 1000.
            chunk_overlap: Ardışık parçalar arasındaki örtüşme karakter sayısı.
                           Bağlamın korunmasına yardımcı olur. Varsayılan: 200.
            separators: Öncelik sırasıyla denenecek ayraç dizelerinin listesi.
                        ``None`` ise ``["\n\n", "\n", ". ", " ", ""]`` kullanılır.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Metni chunk'lara böler.

        Args:
            text: Bölünecek ham metin.

        Returns:
            Boş olmayan metin parçalarının listesi.
        """
        return self._split(text, self.separators)

    def _split(self, text: str, seps: List[str]) -> List[str]:
        """Belirtilen ayraç listesiyle metni yinelemeli olarak böler.

        Metin ``chunk_size``'dan küçükse doğrudan döner. Uygun ayraç
        bulunamazsa sabit adım ve örtüşmeyle keser.

        Args:
            text: Bölünecek metin parçası.
            seps: Sırayla denenecek ayraç listesi.

        Returns:
            Oluşan metin parçalarının listesi.
        """
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
        """Document listesini bölerek yeni Document listesi döner.

        Her belgenin metadata'sı korunur; ek olarak ``"chunk"`` anahtarıyla
        parça indeksi eklenir.

        Args:
            documents: Bölünecek Document nesnelerinin listesi.

        Returns:
            Orijinal metadata + ``chunk`` indeksini taşıyan yeni Document listesi.
        """
        result: List[Document] = []
        for doc in documents:
            for i, chunk in enumerate(self.split_text(doc.page_content)):
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i},
                ))
        return result


class CharacterTextSplitter:
    """Tek bir sabit ayraçla metni bölen basit bölücü.

    ``RecursiveCharacterTextSplitter``'dan farklı olarak yinelemeli
    değildir ve yalnızca bir ayraç kullanır. Örtüşme (chunk_overlap)
    bu sınıfta uygulanmaz.
    """

    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """CharacterTextSplitter oluşturur.

        Args:
            separator: Metnin bölüneceği ayraç dizesi. Varsayılan: ``"\n\n"``.
            chunk_size: Bir parçanın maksimum karakter uzunluğu. Varsayılan: 1000.
            chunk_overlap: Şu an kullanılmıyor; gelecek uyumluluk için tutulur.
        """
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Metni ``separator`` ile böler ve boyutu aşan parçaları birleştirir.

        Ayraçla ayrılan parçalar, ``chunk_size`` sınırını aşmadıkça birleştirilir.
        Sınır aşıldığında mevcut parça kaydedilir ve yeni birikim başlar.

        Args:
            text: Bölünecek ham metin.

        Returns:
            Metin parçalarının listesi.
        """
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
        """Document listesini bölerek yeni Document listesi döner.

        Her belgenin orijinal metadata'sı korunur ve ``"chunk"`` indeksi eklenir.

        Args:
            documents: Bölünecek Document nesnelerinin listesi.

        Returns:
            Parçalanmış Document nesnelerinin listesi.
        """
        result: List[Document] = []
        for doc in documents:
            for i, chunk in enumerate(self.split_text(doc.page_content)):
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i},
                ))
        return result
