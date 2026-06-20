"""
Örnek 3 — Konuşma belleği (Memory)
LangChain: ConversationBufferMemory
Safechain: aynı API
"""
import os
from safechain import (
    Claude,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    LLMChain,
    PromptTemplate,
)

llm = Claude(model="claude-haiku-4-5-20251001", api_key=os.environ["ANTHROPIC_API_KEY"])

# Tüm geçmişi saklayan bellek
memory = ConversationBufferMemory(memory_key="history")

prompt = PromptTemplate.from_template(
    "Önceki konuşma:\n{history}\n\nKullanıcı: {input}\nAsistan:"
)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

print("=== Çok Turlu Konuşma ===")
for soru in [
    "Merhaba, benim adım Mehmet.",
    "Benim adım ne?",
    "Şimdi bir şiir söyle.",
]:
    yanit = chain.run(soru)
    print(f"Soru: {soru}")
    print(f"Yanıt: {yanit}\n")

# Son k konuşmayı saklayan pencereli bellek
print("\n=== Pencereli Bellek (k=2) ===")
window_memory = ConversationBufferWindowMemory(k=2, memory_key="history")
chain2 = LLMChain(llm=llm, prompt=prompt, memory=window_memory)
for soru in ["1+1 kaçtır?", "2+2?", "3+3?", "İlk sorumu hatırlıyor musun?"]:
    yanit = chain2.run(soru)
    print(f"S: {soru}  ->  Y: {yanit}")
