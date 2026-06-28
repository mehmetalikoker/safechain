"""safechain.rag modülü testleri (loaders, splitter, embeddings, vectorstore)."""
import csv
import json
import math
import os
import tempfile
import unittest

from safechain.rag.embeddings import TFIDFEmbeddings
from safechain.rag.loaders import CSVLoader, DirectoryLoader, JSONLoader, TextLoader
from safechain.rag.splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from safechain.rag.vectorstore import InMemoryVectorStore, VectorStoreRetriever, _cosine
from safechain.schema import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_file(content: str, suffix: str = ".txt") -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestTextLoader(unittest.TestCase):
    def test_load_single_document(self):
        path = _tmp_file("Merhaba dünya")
        try:
            docs = TextLoader(path).load()
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].page_content, "Merhaba dünya")
            self.assertEqual(docs[0].metadata["source"], path)
        finally:
            os.unlink(path)

    def test_metadata_has_source(self):
        path = _tmp_file("içerik")
        try:
            doc = TextLoader(path).load()[0]
            self.assertIn("source", doc.metadata)
        finally:
            os.unlink(path)

    def test_encoding_parameter(self):
        path = _tmp_file("özel karakterler: çğıöşü", suffix=".txt")
        try:
            docs = TextLoader(path, encoding="utf-8").load()
            self.assertIn("çğıöşü", docs[0].page_content)
        finally:
            os.unlink(path)


class TestJSONLoader(unittest.TestCase):
    def test_load_list(self):
        path = _tmp_file(json.dumps(["a", "b", "c"]), suffix=".json")
        try:
            docs = JSONLoader(path).load()
            self.assertEqual(len(docs), 3)
            self.assertEqual(docs[0].page_content, "a")
            self.assertEqual(docs[1].metadata["index"], 1)
        finally:
            os.unlink(path)

    def test_load_dict(self):
        path = _tmp_file(json.dumps({"key": "value"}), suffix=".json")
        try:
            docs = JSONLoader(path).load()
            self.assertEqual(len(docs), 1)
            self.assertIn("key", docs[0].page_content)
        finally:
            os.unlink(path)

    def test_list_of_dicts(self):
        data = [{"ad": "Ali"}, {"ad": "Veli"}]
        path = _tmp_file(json.dumps(data), suffix=".json")
        try:
            docs = JSONLoader(path).load()
            self.assertEqual(len(docs), 2)
            self.assertIn("Ali", docs[0].page_content)
        finally:
            os.unlink(path)


class TestCSVLoader(unittest.TestCase):
    def test_load_rows(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["isim", "yaş"])
            writer.writeheader()
            writer.writerow({"isim": "Ali", "yaş": "30"})
            writer.writerow({"isim": "Veli", "yaş": "25"})
            path = f.name
        try:
            docs = CSVLoader(path).load()
            self.assertEqual(len(docs), 2)
            self.assertIn("Ali", docs[0].page_content)
            self.assertIn("isim", docs[0].page_content)
            self.assertEqual(docs[0].metadata["row"], 0)
            self.assertEqual(docs[1].metadata["row"], 1)
        finally:
            os.unlink(path)


class TestDirectoryLoader(unittest.TestCase):
    def test_loads_matching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, content in enumerate(["alfa", "beta", "gama"]):
                with open(os.path.join(tmpdir, f"dosya{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(content)
            with open(os.path.join(tmpdir, "ignore.json"), "w") as f:
                f.write("{}")
            docs = DirectoryLoader(tmpdir, glob="*.txt").load()
            self.assertEqual(len(docs), 3)
            contents = {d.page_content for d in docs}
            self.assertEqual(contents, {"alfa", "beta", "gama"})

    def test_custom_loader_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["x", "y"], f)
            docs = DirectoryLoader(tmpdir, glob="*.json", loader_cls=JSONLoader).load()
            self.assertEqual(len(docs), 2)


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class TestRecursiveCharacterTextSplitter(unittest.TestCase):
    def test_short_text_not_split(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        chunks = splitter.split_text("kısa metin")
        self.assertEqual(chunks, ["kısa metin"])

    def test_splits_on_double_newline(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        text = "Birinci paragraf.\n\nİkinci paragraf."
        chunks = splitter.split_text(text)
        self.assertGreater(len(chunks), 1)

    def test_all_content_preserved(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "Kelime " * 100
        chunks = splitter.split_text(text)
        combined = "".join(chunks)
        for word in text.split():
            self.assertIn(word, combined)

    def test_split_documents(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        docs = [Document(page_content="a" * 50, metadata={"source": "test"})]
        result = splitter.split_documents(docs)
        self.assertGreater(len(result), 1)
        for i, doc in enumerate(result):
            self.assertEqual(doc.metadata["source"], "test")
            self.assertEqual(doc.metadata["chunk"], i)

    def test_empty_text(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100)
        chunks = splitter.split_text("")
        self.assertEqual(chunks, [])

    def test_hard_cut_when_no_separator(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=0, separators=[""])
        chunks = splitter.split_text("abcdefghij")
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 5)


class TestCharacterTextSplitter(unittest.TestCase):
    def test_basic_split(self):
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=50)
        text = "Birinci.\n\nİkinci.\n\nÜçüncü."
        chunks = splitter.split_text(text)
        self.assertEqual(len(chunks), 3)

    def test_merges_small_parts(self):
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=30)
        text = "ab\n\ncd"
        chunks = splitter.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "ab\n\ncd")

    def test_split_documents(self):
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=20)
        docs = [Document(page_content="kısa\n\nparçalar", metadata={"id": 1})]
        result = splitter.split_documents(docs)
        for doc in result:
            self.assertEqual(doc.metadata["id"], 1)
            self.assertIn("chunk", doc.metadata)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestTFIDFEmbeddings(unittest.TestCase):
    def setUp(self):
        self.emb = TFIDFEmbeddings()
        self.corpus = [
            "kedi köpek kuş",
            "araba otobüs tren",
            "elma armut kiraz",
        ]

    def test_embed_documents_returns_list(self):
        vectors = self.emb.embed_documents(self.corpus)
        self.assertEqual(len(vectors), 3)

    def test_vector_length_consistent(self):
        vectors = self.emb.embed_documents(self.corpus)
        lengths = {len(v) for v in vectors}
        self.assertEqual(len(lengths), 1)

    def test_embed_query_after_fit(self):
        self.emb.embed_documents(self.corpus)
        vec = self.emb.embed_query("kedi")
        self.assertIsInstance(vec, list)
        self.assertTrue(all(isinstance(x, float) for x in vec))

    def test_embed_query_without_fit_raises(self):
        with self.assertRaises(RuntimeError):
            self.emb.embed_query("test")

    def test_fit_returns_self(self):
        result = self.emb.fit(self.corpus)
        self.assertIs(result, self.emb)

    def test_tokenize(self):
        tokens = TFIDFEmbeddings._tokenize("Merhaba Dünya 123")
        self.assertIn("merhaba", tokens)
        self.assertIn("dünya", tokens)
        self.assertIn("123", tokens)

    def test_similar_text_closer_than_different(self):
        emb = TFIDFEmbeddings()
        corpus = ["kedi ve köpek", "kedi ve kuş", "araba ve tren"]
        vecs = emb.embed_documents(corpus)
        q = emb.embed_query("kedi")

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            ma = math.sqrt(sum(x**2 for x in a))
            mb = math.sqrt(sum(x**2 for x in b))
            return dot / (ma * mb) if ma and mb else 0.0

        sim_kedi_kopek = cos(q, vecs[0])
        sim_araba_tren = cos(q, vecs[2])
        self.assertGreater(sim_kedi_kopek, sim_araba_tren)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestCosine(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(_cosine(v, v), 1.0, places=5)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(_cosine(a, b), 0.0, places=5)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        self.assertEqual(_cosine(a, b), 0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(_cosine(a, b), -1.0, places=5)


class TestInMemoryVectorStore(unittest.TestCase):
    def _store(self, texts):
        emb = TFIDFEmbeddings()
        docs = [Document(page_content=t) for t in texts]
        return InMemoryVectorStore.from_documents(docs, emb)

    def test_from_documents(self):
        store = self._store(["a b c", "d e f", "g h i"])
        self.assertEqual(len(store._documents), 3)
        self.assertEqual(len(store._embeddings), 3)

    def test_from_texts(self):
        emb = TFIDFEmbeddings()
        store = InMemoryVectorStore.from_texts(
            ["metin1", "metin2"],
            emb,
            metadatas=[{"id": 0}, {"id": 1}],
        )
        self.assertEqual(len(store._documents), 2)
        self.assertEqual(store._documents[0].metadata["id"], 0)

    def test_similarity_search_returns_k(self):
        store = self._store(["elma armut", "kedi köpek", "araba tren", "ev bahçe"])
        results = store.similarity_search("elma", k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], Document)

    def test_similarity_search_with_score(self):
        store = self._store(["python programlama", "java programlama", "pişirme tarifleri"])
        results = store.similarity_search_with_score("programlama", k=2)
        self.assertEqual(len(results), 2)
        doc, score = results[0]
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, results[1][1])

    def test_add_documents(self):
        emb = TFIDFEmbeddings()
        store = InMemoryVectorStore(emb)
        store.add_documents([Document(page_content="ilk metin")])
        store.add_documents([Document(page_content="ikinci metin")])
        self.assertEqual(len(store._documents), 2)

    def test_save_and_load(self):
        store = self._store(["merhaba dünya", "python harika"])
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "store.json")))
            emb2 = TFIDFEmbeddings()
            loaded = InMemoryVectorStore.load(tmpdir, emb2)
        self.assertEqual(len(loaded._documents), 2)
        self.assertEqual(loaded._documents[0].page_content, store._documents[0].page_content)

    def test_as_retriever(self):
        store = self._store(["a b", "c d"])
        retriever = store.as_retriever(k=1)
        self.assertIsInstance(retriever, VectorStoreRetriever)
        self.assertEqual(retriever.k, 1)


class TestVectorStoreRetriever(unittest.TestCase):
    def _retriever(self, texts, k=2):
        emb = TFIDFEmbeddings()
        store = InMemoryVectorStore.from_texts(texts, emb)
        return store.as_retriever(k=k)

    def test_get_relevant_documents(self):
        ret = self._retriever(["elma", "armut", "kedi"], k=2)
        docs = ret.get_relevant_documents("elma")
        self.assertEqual(len(docs), 2)

    def test_call_operator(self):
        ret = self._retriever(["a b", "c d", "e f"], k=1)
        docs = ret("a")
        self.assertEqual(len(docs), 1)


if __name__ == "__main__":
    unittest.main()
