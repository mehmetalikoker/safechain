"""
Örnek 5 — RAG (Retrieval-Augmented Generation)
LangChain: TextLoader + RecursiveCharacterTextSplitter + FAISS + RetrievalQA
Safechain: aynı konsept, sıfır bağımlılık
"""
import os
from safechain import (
    Claude,
    ChatPromptTemplate,
    LLMChain,
    RecursiveCharacterTextSplitter,
    TextLoader,
    TFIDFEmbeddings,   # API gerektirmez — stdlib TF-IDF
    # OpenAIEmbeddings  # API ile daha iyi sonuç için alternatif
    InMemoryVectorStore,
)

# --- 1. Doküman hazırla (örnek metin) ---
SAMPLE_TEXT = """
Türkiye Cumhuriyet Merkez Bankası (TCMB), para politikasını belirleyen bağımsız
bir kurumdur. Merkez bankası faiz kararlarını Para Politikası Kurulu (PPK) alır.

Enflasyon, genel fiyat seviyesindeki sürekli artışı ifade eder. TÜFE (Tüketici
Fiyat Endeksi) en yaygın ölçüm aracıdır.

Döviz rezervleri, bir ülkenin merkez bankasının elinde tuttuğu yabancı para
birimi ve altın varlıklarıdır.

Bütçe dengesi, devlet gelir ve giderlerinin farkıdır. Gelirler giderleri aşarsa
bütçe fazlası, tersi durumda bütçe açığı oluşur.
"""

# Geçici dosya yaz
import tempfile, pathlib
tmp = pathlib.Path(tempfile.gettempdir()) / "ekonomi.txt"
tmp.write_text(SAMPLE_TEXT, encoding="utf-8")

# --- 2. Yükle ve parçala ---
loader = TextLoader(str(tmp))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunk oluşturuldu.")

# --- 3. Embedding + VectorStore ---
# Seçenek A: API olmadan TF-IDF (bankaların kısıtlı ortamı için ideal)
embeddings = TFIDFEmbeddings()
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# Seçenek B: OpenAI embedding (daha iyi kalite, API gerekir)
# embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
# vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# --- 4. Retriever ---
retriever = vectorstore.as_retriever(k=3)

# --- 5. RAG chain ---
llm = Claude(model="claude-haiku-4-5-20251001", api_key=os.environ["ANTHROPIC_API_KEY"])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Aşağıdaki bağlamı kullanarak soruyu yanıtla. Bağlamda yoksa 'bilmiyorum' de.\n\nBağlam:\n{context}"),
    ("user", "{soru}"),
])
rag_chain = LLMChain(llm=llm, prompt=rag_prompt, output_key="cevap")

def rag_qa(soru: str) -> str:
    docs = retriever(soru)
    context = "\n\n".join(d.page_content for d in docs)
    return rag_chain.invoke({"soru": soru, "context": context})["cevap"]

# --- 6. Test ---
sorular = [
    "TCMB ne yapar?",
    "Enflasyon nasıl ölçülür?",
    "Bütçe açığı ne zaman oluşur?",
]
for soru in sorular:
    print(f"\nSoru: {soru}")
    print(f"Yanıt: {rag_qa(soru)}")

# --- 7. VectorStore kaydet / yükle ---
vectorstore.save("./rag_store")
loaded = InMemoryVectorStore.load("./rag_store", embeddings)
print(f"\nYüklenen store'dan arama: {loaded.similarity_search('merkez bankası', k=1)[0].page_content[:80]}")
