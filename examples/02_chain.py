"""
Örnek 2 — LLMChain ve SequentialChain
LangChain: LLMChain(llm=..., prompt=...)
Safechain: aynı API
"""
import os
from safechain import (
    Claude,
    ChatPromptTemplate,
    HumanMessage,
    LLMChain,
    PromptTemplate,
    SequentialChain,
    SystemMessage,
)

llm = Claude(model="claude-haiku-4-5-20251001", api_key=os.environ["ANTHROPIC_API_KEY"])

# 1. Basit LLMChain
prompt = PromptTemplate.from_template("{konu} hakkında bir başlık yaz.")
chain = LLMChain(llm=llm, prompt=prompt, output_key="baslik")
result = chain.invoke({"konu": "yapay zeka"})
print("Başlık:", result["baslik"])

# 2. ChatPromptTemplate kullanımı
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen kısa ve öz cevaplar veren bir finans asistanısın."),
    ("user", "{soru}"),
])
chat_chain = LLMChain(llm=llm, prompt=chat_prompt, output_key="cevap")
result2 = chat_chain.invoke({"soru": "Enflasyon nedir?"})
print("Cevap:", result2["cevap"])

# 3. SequentialChain — çıktıları birbirine bağla
ozet_prompt = PromptTemplate.from_template("{metin} metnini 1 cümlede özetle.")
baslik_prompt = PromptTemplate.from_template("{ozet} için akılda kalıcı bir başlık yaz.")

ozet_chain  = LLMChain(llm=llm, prompt=ozet_prompt, output_key="ozet")
baslik_chain = LLMChain(llm=llm, prompt=baslik_prompt, output_key="baslik")

seq = SequentialChain(
    chains=[ozet_chain, baslik_chain],
    input_variables=["metin"],
    output_variables=["ozet", "baslik"],
    verbose=True,
)
cikti = seq.invoke({"metin": "Merkez bankası faiz oranlarını 50 baz puan artırdı..."})
print("Özet:", cikti["ozet"])
print("Başlık:", cikti["baslik"])
