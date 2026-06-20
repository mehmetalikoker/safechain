from __future__ import annotations
import csv
import fnmatch
import json
import os
from typing import List, Optional, Type

from safechain.schema import Document


class TextLoader:
    def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        with open(self.file_path, encoding=self.encoding) as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": self.file_path})]


class JSONLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> List[Document]:
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
    def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
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
    def __init__(
        self,
        path: str,
        glob: str = "*.txt",
        loader_cls: Optional[Type] = None,
    ) -> None:
        self.path = path
        self.glob = glob
        self.loader_cls: Type = loader_cls or TextLoader

    def load(self) -> List[Document]:
        docs: List[Document] = []
        for fname in os.listdir(self.path):
            if fnmatch.fnmatch(fname, self.glob):
                fpath = os.path.join(self.path, fname)
                docs.extend(self.loader_cls(fpath).load())
        return docs
