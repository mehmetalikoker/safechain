"""
safechain — LangChain'in sıfır bağımlılıklı, saf Python implementasyonu.
Sadece Python standart kütüphanesi kullanır (stdlib only).
"""
from safechain.schema import Document, Generation, Message

from safechain.llm import BaseLLM, Claude, OpenAI
from safechain.prompts import (
    AIMessage,
    ChatPromptTemplate,
    HumanMessage,
    PromptTemplate,
    SystemMessage,
)
from safechain.chains import Chain, LLMChain, SequentialChain, SimpleSequentialChain
from safechain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from safechain.tools import Tool, tool
from safechain.agents import AgentExecutor
from safechain.output_parsers import (
    BaseOutputParser,
    CommaSeparatedListOutputParser,
    JSONOutputParser,
    NumberedListOutputParser,
    OutputParserException,
    ParsedChain,
    ResponseSchema,
    StrOutputParser,
    StructuredOutputParser,
)
from safechain.rag import (
    CharacterTextSplitter,
    CSVLoader,
    DirectoryLoader,
    InMemoryVectorStore,
    JSONLoader,
    OpenAIEmbeddings,
    RecursiveCharacterTextSplitter,
    TextLoader,
    TFIDFEmbeddings,
    VectorStoreRetriever,
)

__version__ = "0.1.0"
__all__ = [
    # Schema
    "Message", "Document", "Generation",
    # LLM
    "BaseLLM", "Claude", "OpenAI",
    # Prompts
    "PromptTemplate", "ChatPromptTemplate",
    "SystemMessage", "HumanMessage", "AIMessage",
    # Chains
    "Chain", "LLMChain", "SequentialChain", "SimpleSequentialChain",
    # Memory
    "ConversationBufferMemory", "ConversationBufferWindowMemory",
    # Tools & Agents
    "Tool", "tool", "AgentExecutor",
    # Output Parsers
    "BaseOutputParser", "OutputParserException", "ParsedChain",
    "StrOutputParser", "CommaSeparatedListOutputParser", "NumberedListOutputParser",
    "JSONOutputParser", "ResponseSchema", "StructuredOutputParser",
    # RAG
    "TextLoader", "JSONLoader", "CSVLoader", "DirectoryLoader",
    "CharacterTextSplitter", "RecursiveCharacterTextSplitter",
    "OpenAIEmbeddings", "TFIDFEmbeddings",
    "InMemoryVectorStore", "VectorStoreRetriever",
]
