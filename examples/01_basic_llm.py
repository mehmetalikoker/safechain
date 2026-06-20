"""
Örnek 1 — Temel LLM çağrısı
LangChain: from langchain_anthropic import ChatAnthropic
Safechain: from safechain import Claude
"""
import os
from safechain import Claude, OpenAI, PromptTemplate

# --- Anthropic ---
llm = Claude(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens=512,
)

yanit = llm.predict("Türkiye'nin başkenti neresidir?")
print("Claude:", yanit)

# --- OpenAI ---
# llm = OpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
# yanit = llm.predict("Türkiye'nin başkenti neresidir?")
# print("OpenAI:", yanit)

# --- PromptTemplate ile ---
template = PromptTemplate.from_template(
    "{sehir} şehrinin nüfusu kaçtır? Kısa cevap ver."
)
prompt_str = template.format(sehir="İstanbul")
print("\nPrompt:", prompt_str)
print("Yanıt:", llm.predict(prompt_str))
