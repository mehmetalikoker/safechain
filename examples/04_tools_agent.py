"""
Örnek 4 — Araçlar (Tools) ve Ajan (AgentExecutor)
LangChain: @tool decorator + AgentExecutor
Safechain: aynı API
"""
import math
import os
from safechain import AgentExecutor, Claude, tool


@tool
def topla(a: float, b: float) -> float:
    """İki sayıyı toplar."""
    return a + b


@tool
def karekok(sayi: float) -> float:
    """Bir sayının karekökünü hesaplar."""
    return math.sqrt(sayi)


@tool(name="doviz_kuru", description="USD/TRY kuru döner (mock).")
def doviz_kuru(para_birimi: str) -> str:
    kurlar = {"USD": "32.5", "EUR": "35.1", "GBP": "41.2"}
    return kurlar.get(para_birimi.upper(), "Bilinmeyen para birimi")


llm = Claude(
    model="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

agent = AgentExecutor(
    llm=llm,
    tools=[topla, karekok, doviz_kuru],
    verbose=True,
    system_prompt="Sen matematik ve finans konularında uzman bir asistandsın.",
)

sorular = [
    "144'ün karekökü nedir?",
    "Bugünkü USD/TRY kuru nedir? 1000 dolar kaç TL eder?",
    "3.14 ile 2.71'i topla, sonra toplamın karekökünü al.",
]

for soru in sorular:
    print(f"\nSoru: {soru}")
    print(f"Yanıt: {agent.run(soru)}")
    print("-" * 60)
