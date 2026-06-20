# safechain

**LangChain'in sıfır bağımlılıklı, saf Python implementasyonu.**

Bankalar ve finans kurumları gibi üçüncü taraf kütüphane kullanımının kısıtlı olduğu ortamlar için tasarlandı. Tek bir `pip install` bile gerektirmez — yalnızca Python standart kütüphanesi (stdlib) kullanılır.

---

## İçindekiler

- [Neden safechain?](#neden-safechain)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Modüller](#modüller)
  - [LLM](#llm)
  - [Prompts](#prompts)
  - [Chains](#chains)
  - [Memory](#memory)
  - [Tools & Agents](#tools--agents)
  - [RAG](#rag)
- [Örnekler](#örnekler)
- [LangChain → safechain Karşılaştırması](#langchain--safechain-karşılaştırması)
- [Mimari](#mimari)
- [Geliştirme](#geliştirme)

---

## Neden safechain?

| Sorun | Çözüm |
|---|---|
| LangChain 100+ bağımlılık getirir | safechain = 0 bağımlılık |
| Banka güvenlik politikaları 3rd-party paketleri yasaklar | Sadece Python stdlib kullanılır |
| LangChain audit edilmesi zor | ~1500 satır, tek bir klasörde |
| Versiyon çakışmaları | `pyproject.toml`'da `dependencies = []` |

safechain, LangChain'in **aynı API'sini** sunar; böylece mevcut kod örneklerini minimum değişiklikle adapte edebilirsiniz.

---

## Kurulum

```bash
# Pip ile (geliştirme modu)
pip install -e .

# VEYA — kütüphaneyi doğrudan proje içine kopyala (zero-install)
cp -r safechain/ your_project/
```

> **Sıfır bağımlılık:** `pip install` sadece paketi Python path'e ekler. Herhangi bir dış kütüphane çekilmez.

### Ortam Değişkenleri

```bash
# Anthropic (Claude) için
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI için
export OPENAI_API_KEY="sk-..."
```

---

## Hızlı Başlangıç

```python
from safechain import Claude, PromptTemplate, LLMChain

llm = Claude(model="claude-haiku-4-5-20251001")

prompt = PromptTemplate.from_template("{konu} hakkında kısa bir özet yaz.")
chain = LLMChain(llm=llm, prompt=prompt)

sonuc = chain.run(konu="merkez bankası faiz politikası")
print(sonuc)
```

---

## Modüller

### LLM

LLM'ler `urllib.request` ile doğrudan REST API çağrısı yapar. Hiçbir SDK gerekmez.

#### Claude (Anthropic)

```python
from safechain import Claude

llm = Claude(
    model="claude-sonnet-4-6",      # Varsayılan
    api_key="sk-ant-...",           # Yoksa ANTHROPIC_API_KEY env var'ı
    max_tokens=4096,
    temperature=1.0,
    system="Sen bir finans asistanısın.",  # Opsiyonel sistem mesajı
)

# Basit çağrı
yanit = llm.predict("Enflasyon nedir?")

# Callable olarak
yanit = llm("Enflasyon nedir?")

# Message listesiyle
from safechain import Message
yanit = llm.predict_messages([
    Message(role="system", content="Kısa cevap ver."),
    Message(role="user", content="Enflasyon nedir?"),
])
```

**Mevcut Claude modelleri:**

| Model ID | Açıklama |
|---|---|
| `claude-haiku-4-5-20251001` | Hızlı, ekonomik |
| `claude-sonnet-4-6` | Dengeli (varsayılan) |
| `claude-opus-4-8` | En güçlü |

#### OpenAI

```python
from safechain import OpenAI

llm = OpenAI(
    model="gpt-4o",
    api_key="sk-...",               # Yoksa OPENAI_API_KEY env var'ı
    base_url="https://api.openai.com/v1",  # Uyumlu API'ler için değiştirilebilir
    max_tokens=4096,
    temperature=1.0,
)

yanit = llm.predict("Merkez bankası ne yapar?")
```

> **OpenAI-uyumlu API'ler:** `base_url` parametresiyle Azure OpenAI, local LLM (LM Studio, Ollama) gibi uyumlu endpoint'lere bağlanabilirsiniz.

#### Kendi LLM'ini Yaz

```python
from safechain.llm.base import BaseLLM
from safechain.schema import Generation, Message
from typing import List

class OzelLLM(BaseLLM):
    def generate(self, messages: List[Message], **kwargs) -> Generation:
        # Kendi API çağrın buraya
        return Generation(text="Yanıt")

llm = OzelLLM()
print(llm.predict("Merhaba"))
```

---

### Prompts

#### PromptTemplate

```python
from safechain import PromptTemplate

# Template oluştur
prompt = PromptTemplate.from_template(
    "{musteri_adi} için {urun} hakkında bir rapor yaz."
)

# input_variables otomatik tespit edilir: ["musteri_adi", "urun"]
print(prompt.input_variables)  # ['musteri_adi', 'urun']

# String döndür
metin = prompt.format(musteri_adi="Ahmet Bey", urun="kredi kartı")

# Manuel tanım
prompt2 = PromptTemplate(
    template="{soru}",
    input_variables=["soru"],
)
```

#### ChatPromptTemplate

```python
from safechain import ChatPromptTemplate

# Tuple listesiyle
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen {rol} uzmanısın. Kısa ve net cevap ver."),
    ("user", "{soru}"),
])

# Message nesneleriyle
from safechain import SystemMessage, HumanMessage, AIMessage

prompt2 = ChatPromptTemplate.from_messages([
    SystemMessage("Sen bir finans asistanısın."),
    HumanMessage("{soru}"),
])

# Message listesi döndür
messages = prompt.format_messages(rol="vergi", soru="KDV nedir?")
# -> [Message(role='system', ...), Message(role='user', ...)]
```

#### Pipe Operatörü `|`

```python
# LangChain LCEL benzeri sözdizimi
chain = PromptTemplate.from_template("{soru}") | llm
sonuc = chain.run(soru="Faiz nedir?")
```

---

### Chains

#### LLMChain

En temel yapı taşı: prompt + llm + (opsiyonel) memory.

```python
from safechain import LLMChain, PromptTemplate, Claude

llm = Claude()
prompt = PromptTemplate.from_template("{metin} metnini özetle.")

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_key="ozet",   # Varsayılan: "text"
)

# invoke — tam dict döner
sonuc = chain.invoke({"metin": "Uzun bir rapor metni..."})
print(sonuc["ozet"])

# run — sadece string döner
ozet = chain.run("Uzun bir rapor metni...")

# predict — kwargs ile
ozet = chain.predict(metin="Uzun bir rapor metni...")
```

#### SimpleSequentialChain

Bir chain'in çıktısı sıradaki chain'in girdisi olur.

```python
from safechain import LLMChain, SimpleSequentialChain, PromptTemplate, Claude

llm = Claude()

ceviri_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("{text} metnini Türkçeye çevir."),
    output_key="text",
)
ozet_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("{text} metnini 1 cümlede özetle."),
    output_key="text",
)

zincir = SimpleSequentialChain(
    chains=[ceviri_chain, ozet_chain],
    verbose=True,
)

sonuc = zincir.run("The central bank raised interest rates by 50 basis points.")
print(sonuc)
```

#### SequentialChain

Birden fazla input/output değişkeni destekler.

```python
from safechain import SequentialChain, LLMChain, PromptTemplate, Claude

llm = Claude()

ozet_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("{metin} metnini özetle."),
    output_key="ozet",
)
baslik_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("{ozet} için bir başlık öner."),
    output_key="baslik",
)

seq = SequentialChain(
    chains=[ozet_chain, baslik_chain],
    input_variables=["metin"],
    output_variables=["ozet", "baslik"],
    verbose=True,
)

sonuc = seq.invoke({"metin": "Merkez bankası toplantısında..."})
print(sonuc["ozet"])
print(sonuc["baslik"])
```

#### Özel Chain

```python
from safechain.chains.base import Chain
from typing import Any, Dict, List

class DogrulamaChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["iban"]

    @property
    def output_keys(self) -> List[str]:
        return ["gecerli", "mesaj"]

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        iban = inputs["iban"].replace(" ", "")
        gecerli = iban.startswith("TR") and len(iban) == 26
        return {
            "gecerli": gecerli,
            "mesaj": "IBAN geçerli." if gecerli else "IBAN hatalı.",
        }

chain = DogrulamaChain()
print(chain.invoke({"iban": "TR33 0006 1005 1978 6457 8413 26"}))
```

---

### Memory

#### ConversationBufferMemory

Tüm konuşma geçmişini saklar.

```python
from safechain import (
    Claude, ConversationBufferMemory, LLMChain, PromptTemplate
)

llm = Claude()
memory = ConversationBufferMemory(memory_key="history")

prompt = PromptTemplate.from_template(
    "Geçmiş konuşma:\n{history}\n\nKullanıcı: {input}\nAsistan:"
)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Çok turlu konuşma
chain.run("Benim adım Mehmet.")
chain.run("Adım ne?")     # "Mehmet" hatırlanır
chain.run("Mesleğim ne?") # Bilinmiyor, ama adı biliyor

# Geçmişi görüntüle
print(memory.messages)  # [Message(role='user',...), Message(role='assistant',...)...]

# Sıfırla
memory.clear()
```

#### ConversationBufferWindowMemory

Yalnızca son `k` konuşma turunu saklar. Büyük konuşmalarda token tasarrufu sağlar.

```python
from safechain import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=3,                    # Son 3 tur (6 mesaj)
    memory_key="history",
    human_prefix="Müşteri",
    ai_prefix="Asistan",
)
```

#### Message nesneleri döndürme

```python
memory = ConversationBufferMemory(
    memory_key="messages",
    return_messages=True,  # str yerine List[Message] döner
)
```

---

### Tools & Agents

#### @tool Decorator

```python
from safechain import tool

@tool
def kur_sorgula(para_birimi: str) -> str:
    """Belirtilen para biriminin TL karşılığını döner."""
    kurlar = {"USD": 32.5, "EUR": 35.1, "GBP": 41.2}
    kur = kurlar.get(para_birimi.upper())
    return f"1 {para_birimi} = {kur} TL" if kur else "Bilinmeyen para birimi"

@tool
def musteri_bakiye(hesap_no: str) -> float:
    """Müşteri hesap bakiyesini sorgular."""
    # Gerçek uygulamada veritabanı çağrısı
    return 15750.00

# Tool özellikleri
print(kur_sorgula.name)         # "kur_sorgula"
print(kur_sorgula.description)  # "Belirtilen para biriminin..."
print(kur_sorgula.schema)       # JSON Schema (otomatik üretilir)

# Çağır
print(kur_sorgula(para_birimi="USD"))
```

#### Tool — Özel isim ve şema

```python
from safechain import Tool, tool

# Özel isim/açıklama ile
@tool(name="doviz_kuru", description="Güncel döviz kurunu getirir.")
def get_rate(currency: str) -> str:
    ...

# Manuel Tool nesnesi
hesap_tool = Tool(
    name="hesap_sorgula",
    description="IBAN ile hesap bilgisi getirir.",
    func=lambda iban: {"bakiye": 5000, "para_birimi": "TRY"},
    args_schema={
        "type": "object",
        "properties": {"iban": {"type": "string"}},
        "required": ["iban"],
    },
)
```

#### AgentExecutor

```python
from safechain import AgentExecutor, Claude, tool
import math

@tool
def hesapla(ifade: str) -> float:
    """Matematiksel ifadeyi hesaplar. Örn: '2 + 2 * 3'"""
    # Güvenli eval — sadece sayı ve operatörler
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in ifade):
        raise ValueError("Geçersiz karakter")
    return eval(ifade)  # noqa: S307

@tool
def karekok(sayi: float) -> float:
    """Sayının karekökünü hesaplar."""
    return math.sqrt(sayi)

llm = Claude(model="claude-sonnet-4-6")

agent = AgentExecutor(
    llm=llm,
    tools=[hesapla, karekok],
    max_iterations=10,
    verbose=True,
    system_prompt="Sen bir matematik asistanısın.",
)

# Çalıştır
sonuc = agent.run("144'ün karekökünü hesapla, sonra buna 8 ekle.")
print(sonuc)

# invoke ile tam dict
sonuc_dict = agent.invoke({"input": "3 * 7 + 12 / 4"})
print(sonuc_dict["output"])
```

**Ajan döngüsü:** LLM → araç çağrısı tespit → aracı çalıştır → sonucu LLM'e gönder → tekrar → cevap üret

AgentExecutor, provider'ı (`Claude` / `OpenAI`) otomatik algılar ve ilgili function-calling protokolünü kullanır:
- Anthropic → `tool_use` / `tool_result` content blokları
- OpenAI → `tool_calls` / `tool` role mesajları

---

### RAG

**R**etrieval-**A**ugmented **G**eneration: Dokümanlardan bilgi çekip LLM'e bağlam olarak sunma.

#### Document Loaders

```python
from safechain import TextLoader, JSONLoader, CSVLoader, DirectoryLoader

# Metin dosyası
docs = TextLoader("rapor.txt", encoding="utf-8").load()

# JSON
docs = JSONLoader("veriler.json").load()

# CSV — her satır bir Document
docs = CSVLoader("musteriler.csv").load()

# Klasördeki tüm .txt dosyaları
docs = DirectoryLoader("./docs", glob="*.txt").load()
docs = DirectoryLoader("./docs", glob="*.csv", loader_cls=CSVLoader).load()

# Document yapısı
print(docs[0].page_content)  # Metin içeriği
print(docs[0].metadata)      # {"source": "rapor.txt"}
```

#### Text Splitters

```python
from safechain import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Önerilen — hiyerarşik bölme (\n\n → \n → . → boşluk)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Maksimum karakter sayısı
    chunk_overlap=200,  # Chunk'lar arası örtüşme (bağlam kaybını önler)
)

chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunk oluşturuldu")
print(chunks[0].page_content)
print(chunks[0].metadata)  # {"source": "...", "chunk": 0}

# Basit bölme — sabit ayraç
splitter2 = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50,
)
chunks2 = splitter2.split_text("Paragraf 1...\n\nParagraf 2...")
```

#### Embeddings

**Seçenek A — TFIDFEmbeddings (API gerektirmez, tamamen stdlib)**

```python
from safechain import TFIDFEmbeddings

embeddings = TFIDFEmbeddings()

# Dokümanları embed et (ilk çağrıda fit + transform)
vectors = embeddings.embed_documents(["metin 1", "metin 2", "metin 3"])

# Sorgu embed et (daha önce fit edilmiş olmalı)
sorgu_vec = embeddings.embed_query("arama terimi")
```

> TF-IDF nöral embedding kadar güçlü değildir ama API erişimi olmayan ortamlarda veya prototipleme için idealdir.

**Seçenek B — OpenAIEmbeddings (daha yüksek kalite, urllib ile)**

```python
from safechain import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # veya text-embedding-3-large
    api_key="sk-...",
    dimensions=512,  # Opsiyonel — vektör boyutunu küçültür
)

vectors = embeddings.embed_documents(["metin 1", "metin 2"])
sorgu_vec = embeddings.embed_query("arama terimi")
```

#### InMemoryVectorStore

Saf Python cosine similarity ile vektör araması. NumPy, FAISS veya başka bir kütüphane gerekmez.

```python
from safechain import InMemoryVectorStore, TFIDFEmbeddings, TextLoader, RecursiveCharacterTextSplitter

# Dokümanları hazırla
docs = TextLoader("bilgi_tabani.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)

# VectorStore oluştur
embeddings = TFIDFEmbeddings()
store = InMemoryVectorStore.from_documents(chunks, embeddings)

# Arama
sonuclar = store.similarity_search("faiz oranı", k=3)
for doc in sonuclar:
    print(doc.page_content[:100])

# Skor ile arama
sonuclar_skor = store.similarity_search_with_score("faiz oranı", k=3)
for doc, skor in sonuclar_skor:
    print(f"Skor: {skor:.3f} | {doc.page_content[:80]}")

# Retriever'a dönüştür
retriever = store.as_retriever(k=4)
ilgili_docs = retriever("merkez bankası kararı")

# Kaydet ve yükle (JSON — stdlib)
store.save("./vector_store")
store2 = InMemoryVectorStore.load("./vector_store", embeddings)
```

#### Tam RAG Pipeline

```python
from safechain import (
    Claude, ChatPromptTemplate, LLMChain,
    TextLoader, RecursiveCharacterTextSplitter,
    TFIDFEmbeddings, InMemoryVectorStore,
)

# 1. Dokümanları yükle ve parçala
docs = TextLoader("sirket_politikalari.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)

# 2. VectorStore kur
store = InMemoryVectorStore.from_documents(chunks, TFIDFEmbeddings())
retriever = store.as_retriever(k=4)

# 3. LLM ve prompt
llm = Claude()
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Sadece aşağıdaki bağlamı kullanarak soruyu yanıtla.\n"
     "Bağlamda cevap yoksa 'Bu konuda bilgim yok.' de.\n\n"
     "Bağlam:\n{context}"),
    ("user", "{soru}"),
])
chain = LLMChain(llm=llm, prompt=prompt, output_key="cevap")

# 4. RAG fonksiyonu
def sor(soru: str) -> str:
    docs = retriever(soru)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return chain.invoke({"soru": soru, "context": context})["cevap"]

print(sor("İzin politikası nedir?"))
print(sor("Uzaktan çalışma kuralları nelerdir?"))
```

---

## Örnekler

`examples/` klasöründe çalıştırılabilir örnekler bulunur:

| Dosya | Konu |
|---|---|
| `01_basic_llm.py` | Claude ve OpenAI temel kullanımı |
| `02_chain.py` | LLMChain ve SequentialChain |
| `03_memory.py` | Konuşma belleği (Buffer + Window) |
| `04_tools_agent.py` | Tool decorator ve AgentExecutor |
| `05_rag.py` | Doküman yükleme, embedding, soru-cevap |

```bash
ANTHROPIC_API_KEY=sk-ant-... python examples/01_basic_llm.py
```

---

## LangChain → safechain Karşılaştırması

### LLM

```python
# LangChain
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# safechain — aynı sonuç
from safechain import Claude
llm = Claude(model="claude-sonnet-4-6")
```

### PromptTemplate

```python
# LangChain
from langchain.prompts import PromptTemplate
p = PromptTemplate.from_template("{konu} hakkında yaz.")

# safechain — birebir aynı
from safechain import PromptTemplate
p = PromptTemplate.from_template("{konu} hakkında yaz.")
```

### LLMChain

```python
# LangChain
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# safechain — birebir aynı
from safechain import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

### Memory

```python
# LangChain
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# safechain — birebir aynı
from safechain import ConversationBufferMemory
memory = ConversationBufferMemory()
```

### @tool Decorator

```python
# LangChain
from langchain.tools import tool
@tool
def hesapla(x: int) -> int:
    """Hesaplar."""
    return x * 2

# safechain — birebir aynı
from safechain import tool
@tool
def hesapla(x: int) -> int:
    """Hesaplar."""
    return x * 2
```

### RAG

```python
# LangChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# safechain — aynı API, sıfır bağımlılık
from safechain import TextLoader, RecursiveCharacterTextSplitter, InMemoryVectorStore
```

---

## Mimari

```
safechain/
├── schema.py              # Paylaşılan tipler: Message, Document, Generation
│
├── llm/
│   ├── base.py            # BaseLLM — soyut sınıf
│   ├── anthropic.py       # Claude — urllib.request ile Anthropic REST API
│   └── openai.py          # OpenAI — urllib.request ile OpenAI REST API
│
├── prompts/
│   └── template.py        # PromptTemplate, ChatPromptTemplate,
│                          # SystemMessage, HumanMessage, AIMessage
│
├── chains/
│   ├── base.py            # Chain — soyut sınıf (invoke, run, __or__)
│   ├── llm_chain.py       # LLMChain
│   └── sequential.py      # SimpleSequentialChain, SequentialChain
│
├── memory/
│   └── buffer.py          # ConversationBufferMemory,
│                          # ConversationBufferWindowMemory
│
├── tools/
│   └── base.py            # Tool, @tool decorator, JSON schema inferrer
│
├── agents/
│   └── executor.py        # AgentExecutor — Anthropic + OpenAI uyumlu
│                          # function-calling döngüsü
│
└── rag/
    ├── loaders.py         # TextLoader, JSONLoader, CSVLoader, DirectoryLoader
    ├── splitter.py        # RecursiveCharacterTextSplitter, CharacterTextSplitter
    ├── embeddings.py      # TFIDFEmbeddings (stdlib), OpenAIEmbeddings (urllib)
    └── vectorstore.py     # InMemoryVectorStore (cosine similarity), VectorStoreRetriever
```

### Bağımlılık Grafiği

```
schema.py          (hiçbir safechain modülüne bağlı değil)
    ↑
llm/base.py
    ↑
llm/anthropic.py   llm/openai.py
    ↑                   ↑
prompts/template.py
    ↑
chains/base.py  →  chains/llm_chain.py  →  chains/sequential.py
                         ↑
                   memory/buffer.py
                         ↑
                   tools/base.py
                         ↑
                   agents/executor.py
                         ↑
                   rag/* (loaders, splitter, embeddings, vectorstore)
```

---

## Geliştirme

### Testler

```bash
python -m pytest tests/ -v
```

### Yeni LLM Eklemek

`safechain/llm/` altında `BaseLLM` miras alan bir sınıf yaz:

```python
from safechain.llm.base import BaseLLM
from safechain.schema import Generation, Message
from typing import List

class BedrockLLM(BaseLLM):
    def generate(self, messages: List[Message], **kwargs) -> Generation:
        # AWS Bedrock çağrısı (urllib ile)
        ...
        return Generation(text=yanit)
```

### Yeni Loader Eklemek

```python
from safechain.schema import Document
from typing import List

class PDFLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> List[Document]:
        # stdlib ile PDF parse (pdfminer yerine saf Python)
        ...
        return [Document(page_content=metin, metadata={"source": self.file_path})]
```

---

## Lisans

MIT

---

## Katkıda Bulunanlar

- [mehmetalikoker](https://github.com/mehmetalikoker)
