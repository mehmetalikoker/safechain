from __future__ import annotations
import csv
import fnmatch
import json
import os
from typing import List, Optional, Type

from safechain.schema import Document


class TextLoader:
    """Düz metin dosyalarını tek bir Document olarak yükler."""

    def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
        """TextLoader oluşturur.

        Args:
            file_path: Okunacak .txt (veya herhangi bir metin) dosyasının yolu.
            encoding: Dosya karakter kodlaması. Varsayılan: "utf-8".
        """
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Dosyayı okur ve tek elemanlı Document listesi döner.

        Returns:
            ``page_content`` dosya metnini, ``metadata["source"]`` dosya
            yolunu içeren tek elemanlı liste.
        """
        with open(self.file_path, encoding=self.encoding) as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": self.file_path})]


class JSONLoader:
    """JSON dosyalarını Document listesine dönüştürerek yükler."""

    def __init__(self, file_path: str) -> None:
        """JSONLoader oluşturur.

        Args:
            file_path: Okunacak .json dosyasının yolu.
        """
        self.file_path = file_path

    def load(self) -> List[Document]:
        """JSON dosyasını ayrıştırır ve Document listesi döner.

        Kök nesne bir liste ise her eleman ayrı bir Document olarak döner;
        string elemanlar doğrudan, diğerleri JSON metni olarak serileştirilir.
        Kök nesne bir dict ise tüm içerik tek Document olarak döner.

        Returns:
            Yüklenen Document nesnelerinin listesi.
        """
        with open(self.file_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [
                Document(
                    page_content=item if isinstance(item, str) else json.dumps(item, ensure_ascii=False),
                    metadata={"source": self.file_path, "index": i},
                )
                for i, item in enumerate(data)
            ]
        return [
            Document(
                page_content=json.dumps(data, ensure_ascii=False),
                metadata={"source": self.file_path},
            )
        ]


class CSVLoader:
    """CSV dosyalarını satır başına bir Document olarak yükler."""

    def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
        """CSVLoader oluşturur.

        Args:
            file_path: Okunacak .csv dosyasının yolu.
            encoding: Dosya karakter kodlaması. Varsayılan: "utf-8".
        """
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """CSV dosyasını okur; her satırı ayrı bir Document olarak döner.

        Her Document'ın içeriği ``sütun: değer`` biçiminde satır satır
        oluşturulur. Metadata'da kaynak dosya yolu ve satır indeksi saklanır.

        Returns:
            Her CSV satırı için bir Document içeren liste.
        """
        docs: List[Document] = []
        with open(self.file_path, newline="", encoding=self.encoding) as f:
            for i, row in enumerate(csv.DictReader(f)):
                content = "\n".join(f"{k}: {v}" for k, v in row.items())
                docs.append(Document(
                    page_content=content,
                    metadata={"source": self.file_path, "row": i},
                ))
        return docs


class DirectoryLoader:
    """Bir dizindeki eşleşen dosyaları toplu olarak yükler."""

    def __init__(
        self,
        path: str,
        glob: str = "*.txt",
        loader_cls: Optional[Type] = None,
    ) -> None:
        """DirectoryLoader oluşturur.

        Args:
            path: Taranacak dizinin yolu.
            glob: Eşleştirilecek dosya deseni (örn. ``"*.txt"``, ``"*.json"``).
                  Varsayılan: "*.txt".
            loader_cls: Her dosya için kullanılacak yükleyici sınıfı.
                        ``None`` ise ``TextLoader`` kullanılır.
        """
        self.path = path
        self.glob = glob
        self.loader_cls: Type = loader_cls or TextLoader

    def load(self) -> List[Document]:
        """Dizindeki glob desenine uyan tüm dosyaları yükler.

        Her dosya için belirtilen yükleyici sınıfı örneklenir ve
        ``load()`` metodu çağrılır. Tüm dökümanlar tek listede birleştirilir.

        Returns:
            Dizindeki tüm eşleşen dosyalardan yüklenen Document listesi.
        """
        docs: List[Document] = []
        for fname in os.listdir(self.path):
            if fnmatch.fnmatch(fname, self.glob):
                fpath = os.path.join(self.path, fname)
                docs.extend(self.loader_cls(fpath).load())
        return docs
