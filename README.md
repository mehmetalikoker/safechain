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
  - [Output Parsers](#output-parsers)
  - [Caching](#caching)
  - [Tools & Agents](#tools--agents)
  - [RAG](#rag)
- [Örnekler](#örnekler)
- [LangChain → safechain Karşılaştırması](#langchain--safechain-karşılaştırması)
- [Mimari](#mimari)
- [Geliştirme](#geliştirme)
- [Lisans](#lisans)

---

## Neden safechain?

| Sorun | Çözüm |
|---|---|
| LangChain 100+ bağımlılık getirir | safechain = 0 bağımlılık |
| Banka güvenlik politikaları 3rd-party paketleri yasaklar | Sadece Python stdlib kullanılır |
| LangChain audit edilmesi zor | ~2000 satır, tek bir klasörde |
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
from safechain import Claude, PromptTemplate, LLMChain, JSONOutputParser

llm = Claude(model="claude-haiku-4-5-20251001")

# Pipe operatörü ile tam zincir: prompt | llm | parser
chain = PromptTemplate.from_template("{konu} için JSON özet üret.") | llm | JSONOutputParser()

sonuc = chain.run(konu="merkez bankası faiz politikası")
print(sonuc)  # → {"özet": "...", "etki": "..."}
```

---

## Modüller

### LLM

LLM'ler `urllib.request` ile doğrudan REST API çağrısı yapar. Hiçbir SDK gerekmez.

#### Claude (Anthropic)

```python
from safechain import Claude, InMemoryCache

llm = Claude(
    model="claude-sonnet-4-6",      # Varsayılan
    api_key="sk-ant-...",           # Yoksa ANTHROPIC_API_KEY env var'ı
    max_tokens=4096,
    temperature=1.0,
    system="Sen bir finans asistanısın.",  # Opsiyonel sistem mesajı
    cache=InMemoryCache(),          # Opsiyonel — aynı prompt için API tekrar çağrılmaz
)

yanit = llm.predict("Enflasyon nedir?")
yanit = llm("Enflasyon nedir?")           # Callable kullanımı
```

**Mevcut Claude modelleri:**

| Model ID | Açıklama |
|---|---|
| `claude-haiku-4-5-20251001` | Hızlı, ekonomik |
| `claude-sonnet-4-6` | Dengeli (varsayılan) |
| `claude-opus-4-8` | En güçlü |

#### OpenAI

```python
from safechain import OpenAI, SQLiteCache

llm = OpenAI(
    model="gpt-4o",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",  # Uyumlu API'ler için değiştirilebilir
    max_tokens=4096,
    temperature=1.0,
    cache=SQLiteCache("cache.db", ttl=3600),  # 1 saatlik kalıcı cache
)
```

> **OpenAI-uyumlu API'ler:** `base_url` parametresiyle Azure OpenAI, LM Studio, Ollama gibi uyumlu endpoint'lere bağlanabilirsiniz.

#### Kendi LLM'ini Yaz

```python
from safechain.llm.base import BaseLLM
from safechain.schema import Generation, Message
from typing import List, Any

class OzelLLM(BaseLLM):
    def _generate(self, messages: List[Message], **kwargs: Any) -> Generation:
        # Kendi API çağrın buraya — cache kontrolü BaseLLM tarafından yapılır
        return Generation(text="Yanıt")

llm = OzelLLM()
print(llm.predict("Merhaba"))
```

---

### Prompts

#### PromptTemplate

```python
from safechain import PromptTemplate

prompt = PromptTemplate.from_template("{musteri_adi} için {urun} hakkında rapor yaz.")

print(prompt.input_variables)   # ['musteri_adi', 'urun']
metin = prompt.format(musteri_adi="Ahmet Bey", urun="kredi kartı")
```

#### ChatPromptTemplate

```python
from safechain import ChatPromptTemplate, SystemMessage, HumanMessage

prompt = ChatPromptTemplate.from_messages([
    SystemMessage("Sen {rol} uzmanısın."),
    HumanMessage("{soru}"),
])

messages = prompt.format_messages(rol="vergi", soru="KDV nedir?")
# → [Message(role='system', ...), Message(role='user', ...)]
```

#### Pipe Operatörü `|`

```python
# Prompt → LLM → Parser zinciri
from safechain import PromptTemplate, Claude, StrOutputParser

chain = PromptTemplate.from_template("{soru}") | Claude() | StrOutputParser()
sonuc = chain.run(soru="Faiz nedir?")  # → str
```

---

### Chains

#### LLMChain

```python
from safechain import LLMChain, PromptTemplate, Claude

chain = LLMChain(
    llm=Claude(),
    prompt=PromptTemplate.from_template("{metin} metnini özetle."),
    output_key="ozet",
)

sonuc = chain.invoke({"metin": "Uzun bir rapor..."})  # → {"ozet": "..."}
ozet  = chain.run("Uzun bir rapor...")                 # → str
ozet  = chain.predict(metin="Uzun bir rapor...")       # → str
```

#### SimpleSequentialChain

```python
from safechain import SimpleSequentialChain, LLMChain, PromptTemplate, Claude

llm = Claude()

zincir = SimpleSequentialChain(chains=[
    LLMChain(llm=llm, prompt=PromptTemplate.from_template("{text} metnini çevir."), output_key="text"),
    LLMChain(llm=llm, prompt=PromptTemplate.from_template("{text} metnini özetle."), output_key="text"),
], verbose=True)

sonuc = zincir.run("The central bank raised rates.")
```

#### SequentialChain

```python
from safechain import SequentialChain, LLMChain, PromptTemplate, Claude

llm = Claude()

seq = SequentialChain(
    chains=[
        LLMChain(llm=llm, prompt=PromptTemplate.from_template("{metin} özetle."), output_key="ozet"),
        LLMChain(llm=llm, prompt=PromptTemplate.from_template("{ozet} için başlık öner."), output_key="baslik"),
    ],
    input_variables=["metin"],
    output_variables=["ozet", "baslik"],
)

sonuc = seq.invoke({"metin": "Uzun metin..."})
```

---

### Memory

#### ConversationBufferMemory

```python
from safechain import Claude, ConversationBufferMemory, LLMChain, PromptTemplate

memory = ConversationBufferMemory(memory_key="history")

chain = LLMChain(
    llm=Claude(),
    prompt=PromptTemplate.from_template("Geçmiş:\n{history}\n\nKullanıcı: {input}\nAsistan:"),
    memory=memory,
)

chain.run("Benim adım Mehmet.")
chain.run("Adım ne?")     # "Mehmet" hatırlanır

memory.clear()            # Geçmişi sıfırla
```

#### ConversationBufferWindowMemory

```python
from safechain import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=3,                     # Son 3 tur (6 mesaj)
    memory_key="history",
    human_prefix="Müşteri",
    ai_prefix="Asistan",
)
```

---

### Output Parsers

LLM'in ham metin çıktısını yapılandırılmış Python nesnelerine dönüştürür.
`|` operatörü ile zincirin sonuna bağlanır.

#### StrOutputParser

```python
from safechain import PromptTemplate, Claude, StrOutputParser

chain = PromptTemplate.from_template("{soru}") | Claude() | StrOutputParser()
sonuc = chain.run(soru="Faiz nedir?")
# → "Faiz, ödünç alınan paranın bedeli..."  (str)
```

#### CommaSeparatedListOutputParser

```python
from safechain import CommaSeparatedListOutputParser, PromptTemplate, Claude, LLMChain

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate.from_template(
    "{konu} ile ilgili 5 anahtar kavramı virgülle sırayla yaz.\n{format_instructions}"
)

chain = LLMChain(llm=Claude(), prompt=prompt)
cikti = chain.run(
    konu="merkez bankası",
    format_instructions=parser.get_format_instructions(),
)
liste = parser.parse(cikti)
# → ["faiz", "enflasyon", "rezerv", "para politikası", "likidite"]

# Pipe ile daha kısa:
chain2 = PromptTemplate.from_template("{konu} için 5 kavram, virgülle.") | Claude() | parser
liste2 = chain2.run(konu="merkez bankası")
```

#### NumberedListOutputParser

```python
from safechain import NumberedListOutputParser

parser = NumberedListOutputParser()

# 1. Madde, 1) Madde, 1- Madde formatlarını tanır
liste = parser.parse("1. Birinci\n2. İkinci\n3. Üçüncü")
# → ["Birinci", "İkinci", "Üçüncü"]
```

#### JSONOutputParser

```python
from safechain import JSONOutputParser, PromptTemplate, Claude

parser = JSONOutputParser()

chain = (
    PromptTemplate.from_template(
        "{metin} için JSON analiz üret.\n{format_instructions}"
    )
    | Claude()
    | parser
)

sonuc = chain.run(
    metin="Merkez bankası faizi 250 baz puan artırdı.",
    format_instructions=parser.get_format_instructions(),
)
# → {"karar": "artış", "miktar": "250 baz puan", "etki": "..."}
```

Hem düz JSON hem ````json ... ` ``` `` markdown bloğunu destekler.

#### StructuredOutputParser

```python
from safechain import StructuredOutputParser, ResponseSchema, LLMChain, PromptTemplate, Claude

schemas = [
    ResponseSchema(name="özet",   description="Metnin kısa özeti"),
    ResponseSchema(name="dil",    description="Metnin dili"),
    ResponseSchema(name="puan",   description="Kalite puanı (1-10)", type="number"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)

chain = LLMChain(
    llm=Claude(),
    prompt=PromptTemplate.from_template(
        "{metin}\n\n{format_instructions}"
    ),
)

cikti = chain.run(
    metin="Merkez bankası faizi sabit tuttu.",
    format_instructions=parser.get_format_instructions(),
)

sonuc = parser.parse(cikti)
# → {"özet": "...", "dil": "Türkçe", "puan": "8"}
```

---

### Caching

Aynı prompt için API'yi tekrar çağırmayı önler. Token maliyetini ve gecikmeyi düşürür.

#### InMemoryCache

Process ömrü boyunca geçerlidir; yeniden başlatınca sıfırlanır.

```python
from safechain import Claude, InMemoryCache

cache = InMemoryCache()
llm = Claude(api_key="...", cache=cache)

llm.predict("Faiz nedir?")   # API çağrısı yapılır
llm.predict("Faiz nedir?")   # Cache'den döner, API çağrılmaz

print(len(cache))  # 1
cache.clear()
```

#### SQLiteCache

Disk tabanlı, kalıcı cache. Uygulama yeniden başlatılsa bile geçerliliğini korur.

```python
from safechain import Claude, SQLiteCache

llm = Claude(
    api_key="...",
    cache=SQLiteCache(
        db_path="llm_cache.db",  # Dosya yolu
        ttl=3600,                # Saniye cinsinden geçerlilik süresi (opsiyonel)
    ),
)

llm.predict("Enflasyon nedir?")  # İlk çağrı — API
llm.predict("Enflasyon nedir?")  # Sonraki çağrılar — disk'ten

# Temizle
cache = SQLiteCache("llm_cache.db")
cache.clear()
cache.close()
```

#### Birden Fazla LLM, Tek Cache

```python
from safechain import Claude, OpenAI, InMemoryCache

cache = InMemoryCache()
claude = Claude(api_key="...", cache=cache)
gpt    = OpenAI(api_key="...", cache=cache)
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

@tool(name="doviz_kuru", description="Güncel döviz kurunu getirir.")
def get_rate(currency: str) -> str:
    ...

print(kur_sorgula.name)    # "kur_sorgula"
print(kur_sorgula.schema)  # JSON Schema (otomatik üretilir)
print(kur_sorgula(para_birimi="USD"))
```

#### AgentExecutor

```python
from safechain import AgentExecutor, Claude, tool
import math

@tool
def hesapla(ifade: str) -> float:
    """Matematiksel ifadeyi hesaplar. Örn: '2 + 2 * 3'"""
    return eval(ifade)  # noqa: S307

@tool
def karekok(sayi: float) -> float:
    """Sayının karekökünü hesaplar."""
    return math.sqrt(sayi)

agent = AgentExecutor(
    llm=Claude(),
    tools=[hesapla, karekok],
    max_iterations=10,
    verbose=True,
    system_prompt="Sen bir matematik asistanısın.",
)

sonuc = agent.run("144'ün karekökünü hesapla, sonra buna 8 ekle.")
```

**Provider otomatik algılanır:**
- Claude → Anthropic `tool_use` / `tool_result` protokolü
- OpenAI → `tool_calls` / `tool` role protokolü

---

### RAG

**R**etrieval-**A**ugmented **G**eneration: Dokümanlardan bilgi çekip LLM'e bağlam olarak sunma.

#### Document Loaders

```python
from safechain import TextLoader, JSONLoader, CSVLoader, DirectoryLoader

docs = TextLoader("rapor.txt").load()
docs = JSONLoader("veriler.json").load()
docs = CSVLoader("musteriler.csv").load()
docs = DirectoryLoader("./docs", glob="*.txt").load()
docs = DirectoryLoader("./docs", glob="*.csv", loader_cls=CSVLoader).load()
```

#### Text Splitters

```python
from safechain import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
# chunks[0].metadata → {"source": "rapor.txt", "chunk": 0}
```

#### Embeddings

```python
# Seçenek A — API gerektirmez (stdlib TF-IDF)
from safechain import TFIDFEmbeddings
embeddings = TFIDFEmbeddings()

# Seçenek B — OpenAI (urllib ile, daha yüksek kalite)
from safechain import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-...")
```

#### InMemoryVectorStore

```python
from safechain import InMemoryVectorStore, TFIDFEmbeddings

store = InMemoryVectorStore.from_documents(chunks, TFIDFEmbeddings())

# Arama
sonuclar = store.similarity_search("faiz oranı", k=3)
sonuclar_skor = store.similarity_search_with_score("faiz oranı", k=3)

# Kaydet / Yükle (JSON, stdlib)
store.save("./vector_store")
store2 = InMemoryVectorStore.load("./vector_store", embeddings)

# Retriever'a dönüştür
retriever = store.as_retriever(k=4)
docs = retriever("merkez bankası kararı")
```

#### Tam RAG Pipeline

```python
from safechain import (
    Claude, ChatPromptTemplate, LLMChain,
    TextLoader, RecursiveCharacterTextSplitter,
    TFIDFEmbeddings, InMemoryVectorStore, InMemoryCache,
)

# 1. Dokümanları hazırla
docs   = TextLoader("politikalar.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)

# 2. VectorStore
store     = InMemoryVectorStore.from_documents(chunks, TFIDFEmbeddings())
retriever = store.as_retriever(k=4)

# 3. LLM (cache ile)
llm = Claude(cache=InMemoryCache())

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sadece verilen bağlamı kullanarak yanıtla.\n\nBağlam:\n{context}"),
    ("user", "{soru}"),
])
chain = LLMChain(llm=llm, prompt=prompt, output_key="cevap")

# 4. RAG fonksiyonu
def sor(soru: str) -> str:
    docs    = retriever(soru)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return chain.invoke({"soru": soru, "context": context})["cevap"]

print(sor("İzin politikası nedir?"))
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

```python
# LLM
from langchain_anthropic import ChatAnthropic        # LangChain
from safechain import Claude                          # safechain

# PromptTemplate
from langchain.prompts import PromptTemplate         # LangChain
from safechain import PromptTemplate                  # safechain — birebir aynı

# Chain
from langchain.chains import LLMChain                # LangChain
from safechain import LLMChain                        # safechain — birebir aynı

# Memory
from langchain.memory import ConversationBufferMemory # LangChain
from safechain import ConversationBufferMemory        # safechain — birebir aynı

# Output Parsers
from langchain.output_parsers import JSONOutputParser # LangChain
from safechain import JSONOutputParser                 # safechain — birebir aynı

# Caching
from langchain.cache import InMemoryCache             # LangChain
from safechain import InMemoryCache                   # safechain — birebir aynı

# @tool
from langchain.tools import tool                      # LangChain
from safechain import tool                             # safechain — birebir aynı

# RAG
from langchain_community.vectorstores import FAISS   # LangChain (bağımlılık!)
from safechain import InMemoryVectorStore             # safechain — sıfır bağımlılık
```

---

## Mimari

```
safechain/
├── schema.py                  # Paylaşılan tipler: Message, Document, Generation
│
├── llm/
│   ├── base.py                # BaseLLM — şablon metot (cache + _generate)
│   ├── anthropic.py           # Claude — Anthropic REST API (urllib)
│   └── openai.py              # OpenAI — OpenAI REST API (urllib)
│
├── prompts/
│   └── template.py            # PromptTemplate, ChatPromptTemplate,
│                              # SystemMessage, HumanMessage, AIMessage
│
├── chains/
│   ├── base.py                # Chain — soyut sınıf (invoke, run, __or__)
│   ├── llm_chain.py           # LLMChain
│   └── sequential.py          # SimpleSequentialChain, SequentialChain
│
├── memory/
│   └── buffer.py              # ConversationBufferMemory,
│                              # ConversationBufferWindowMemory
│
├── output_parsers/
│   ├── base.py                # BaseOutputParser, ParsedChain
│   ├── simple.py              # StrOutputParser, CommaSeparatedListOutputParser,
│   │                          # NumberedListOutputParser
│   ├── json_parser.py         # JSONOutputParser
│   └── structured.py          # ResponseSchema, StructuredOutputParser
│
├── cache/
│   ├── base.py                # BaseCache, _cache_key (SHA-256)
│   ├── memory.py              # InMemoryCache
│   └── sqlite.py              # SQLiteCache (stdlib sqlite3, TTL destekli)
│
├── tools/
│   └── base.py                # Tool, @tool decorator, JSON schema inferrer
│
├── agents/
│   └── executor.py            # AgentExecutor — Anthropic + OpenAI uyumlu
│
└── rag/
    ├── loaders.py             # TextLoader, JSONLoader, CSVLoader, DirectoryLoader
    ├── splitter.py            # RecursiveCharacterTextSplitter, CharacterTextSplitter
    ├── embeddings.py          # TFIDFEmbeddings (stdlib), OpenAIEmbeddings (urllib)
    └── vectorstore.py         # InMemoryVectorStore, VectorStoreRetriever
```

### `|` Operatörü Akışı

```
PromptTemplate | LLM            → LLMChain
PromptTemplate | LLM | Parser   → ParsedChain
Chain          | Chain          → SimpleSequentialChain
Chain          | Parser         → ParsedChain
```

### Cache Akışı (Şablon Metot)

```
llm.generate(messages)
    └─ cache.lookup(key)   → HIT  → Generation döner (API çağrılmaz)
                           → MISS → _generate(messages) → cache.update(key, result)
```

---

## Geliştirme

### Testleri Çalıştır

```bash
python -m pytest tests/ -v
# 221 test, ~0.5 saniye (API çağrısı yok — tümü mock)
```

### Yeni LLM Eklemek

```python
from safechain.llm.base import BaseLLM
from safechain.schema import Generation, Message
from typing import Any, List

class BedrockLLM(BaseLLM):
    def __init__(self, model: str, cache=None):
        super().__init__(cache=cache)   # cache desteği otomatik gelir
        self.model = model

    def _generate(self, messages: List[Message], **kwargs: Any) -> Generation:
        # AWS Bedrock çağrısı (urllib ile)
        ...
        return Generation(text=yanit)
```

### Yeni Output Parser Eklemek

```python
from safechain.output_parsers.base import BaseOutputParser

class BulletListParser(BaseOutputParser[list]):
    def parse(self, text: str) -> list:
        return [line.lstrip("•- ").strip() for line in text.splitlines() if line.strip()]

    def get_format_instructions(self) -> str:
        return "Yanıtını madde işaretli liste olarak ver (• veya -)."
```

---

## Lisans

MIT — Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

---

## Katkıda Bulunanlar

- [mehmetalikoker](https://github.com/mehmetalikoker)
