<div align="center">
  <img src="https://raw.githubusercontent.com/SynapseKit/SynapseKit/main/assets/banner.svg" alt="SynapseKit" width="100%"/>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/synapsekit?color=22c55e&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/synapsekit/)
[![Python](https://img.shields.io/badge/python-3.10%2B-22c55e?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-22c55e)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-3195%20passing-22c55e?logo=pytest&logoColor=white)]()
[![Downloads](https://img.shields.io/pypi/dm/synapsekit?color=22c55e&logo=pypi&logoColor=white)](https://pypistats.org/packages/synapsekit)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/synapsekit?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/synapsekit)
[![Docs](https://img.shields.io/badge/docs-online-22c55e?logo=readthedocs&logoColor=white)](https://synapsekit.github.io/synapsekit-docs/)
[![Discord](https://img.shields.io/discord/1488136255597182988?logo=discord&logoColor=white)](https://discord.gg/PSuAXHRywJ)

**[Documentation](https://synapsekit.github.io/synapsekit-docs/) · [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart) · [API Reference](https://synapsekit.github.io/synapsekit-docs/docs/api/llm) · [Changelog](CHANGELOG.md) · [Discord](https://discord.gg/PSuAXHRywJ) · [Report a Bug](https://github.com/SynapseKit/SynapseKit/issues/new?template=bug_report.yml)**

</div>

---

**Build production LLM apps with 2 dependencies.**
Async-native RAG, Agents, and Graph workflows — no magic, no SaaS, no bloat.

> *"LangChain for people who hate LangChain."*

SynapseKit is the minimal, async-first Python framework for LLM applications. 34 providers · 48+ tools · 64 loaders · 22 vector stores. Every abstraction is plain Python you can read, debug, and extend. No hidden chains. No global state. No lock-in.

---

<div align="center">

<table>
<tr>
<td align="center" width="33%">
<h3>⚡ Async-native</h3>
Every API is <code>async/await</code> first.<br/>Sync wrappers for scripts and notebooks.<br/>No event loop surprises.
</td>
<td align="center" width="33%">
<h3>🌊 Streaming-first</h3>
Token-level streaming is the default,<br/>not an afterthought.<br/>Works across all providers.
</td>
<td align="center" width="33%">
<h3>🪶 Minimal footprint</h3>
2 hard dependencies: <code>numpy</code> + <code>rank-bm25</code>.<br/>Everything else is optional.<br/>Install only what you use.
</td>
</tr>
<tr>
<td align="center" width="33%">
<h3>🔌 One interface</h3>
34 LLM providers and 22 vector stores<br/>behind the same API.<br/>Swap without rewriting.
</td>
<td align="center" width="33%">
<h3>🧩 Composable</h3>
RAG pipelines, agents, and graph nodes<br/>are interchangeable.<br/>Wrap anything as anything.
</td>
<td align="center" width="33%">
<h3>🔍 Transparent</h3>
No hidden chains.<br/>Every step is plain Python<br/>you can read and override.
</td>
</tr>
</table>

</div>

---

## 10-Line Agent Example

```python
from synapsekit import agent, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

my_agent = agent(
    model="gpt-4o-mini",
    api_key="sk-...",
    tools=[get_weather],
)

print(my_agent.run("What's the weather in Tokyo?"))
```

---

## SynapseKit vs LangChain vs LlamaIndex

<div align="center">

| | SynapseKit | LangChain | LlamaIndex |
|---|---|---|---|
| Hard dependencies | **2** | 50+ | 20+ |
| Install size | **~5 MB** | ~200 MB+ | ~100 MB+ |
| Async-native | **✅ Default** | ⚠️ Partial | ⚠️ Partial |
| Cost tracking | **✅ Built-in** | ❌ LangSmith (SaaS) | ❌ No |
| Evaluation | **✅ CLI + GitHub Action** | ❌ LangSmith (SaaS) | ✅ Built-in |
| Graph workflows | **✅ Built-in** | ✅ LangGraph (separate pkg) | ❌ No |
| LLM providers | **34** | 38+ | 20+ |
| Stack traces | **Your code** | Framework internals | Framework internals |

</div>

LangChain has more raw integrations and more tutorials. That's not what SynapseKit is optimizing for. SynapseKit is optimizing for the engineer who needs to ship, debug, and maintain an LLM feature in production — where readable code, predictable async behavior, and no surprise SaaS bills actually matter.

---

## Who is it for?

SynapseKit is for Python developers who want to ship LLM features without fighting their framework.

- **Burned LangChain users** — hit a wall with debugging, dependency hell, or version churn and want full control back
- **Async backend engineers** — building FastAPI services where LangChain's sync-first model feels bolted on
- **Cost-conscious teams** — startups and teams who don't want a LangSmith subscription for basic observability
- **ML engineers** — building RAG or agent pipelines who need full control over retrieval, prompting, and tool use

---

## What it covers

<div align="center">

<table>
<tr>
<td width="50%">

**🗂 RAG Pipelines**<br/>
Retrieval-augmented generation with streaming, BM25 reranking, conversation memory, and token tracing. Load from PDFs, URLs, CSVs, HTML, directories, and more.

</td>
<td width="50%">

**🤖 Agents**<br/>
ReAct loop (any LLM) and native function calling (OpenAI / Anthropic / Gemini / Mistral). 48 built-in tools including calculator, Python REPL, code interpreter, web search, SQL, HTTP, shell, Twilio, arxiv, pubmed, wolfram, wikipedia, and more. Fully extensible.

</td>
</tr>
<tr>
<td width="50%">

**🔀 Graph Workflows**<br/>
DAG-based async pipelines. Nodes run in waves — parallel nodes execute concurrently. Conditional routing, typed state with reducers, fan-out/fan-in, SSE streaming, event callbacks, human-in-the-loop, checkpointing, and Mermaid export.

</td>
<td width="50%">

**🧠 LLM Providers**<br/>
OpenAI, Anthropic, Ollama, Gemini, Cohere, Mistral, Bedrock, Azure OpenAI, Groq, DeepSeek, OpenRouter, Together, Fireworks, Cerebras, Cloudflare, Moonshot, Perplexity, Vertex AI, Zhipu, AI21 Labs, Databricks, Baidu ERNIE, llama.cpp, LM Studio, Minimax, Aleph Alpha, Hugging Face, SambaNova, xAI, NovitaAI, Writer — all behind one interface. Auto-detected from the model name. Swap without rewriting.

</td>
</tr>
<tr>
<td width="50%">

**🗄 Vector Stores**<br/>
InMemory (built-in, `.npz` persistence), ChromaDB, FAISS, Qdrant, Pinecone, Weaviate, PGVector, Milvus, LanceDB. One interface for all 9 backends.

</td>
<td width="50%">

**🔧 Utilities**<br/>
Output parsers (JSON, Pydantic, List), prompt templates (standard, chat, few-shot), token tracing with cost estimation.

</td>
</tr>
<tr>
<td width="50%" colspan="2">

**🧪 EvalCI — LLM Quality Gates**<br/>
GitHub Action that runs `@eval_case` suites on every PR and blocks merge if quality drops. No infrastructure, 2-minute setup. Score, cost, and latency tracked per case. Works with any LLM provider. → [GitHub Marketplace](https://github.com/marketplace/actions/evalci-by-synapsekit) · [Docs](https://synapsekit.github.io/synapsekit-docs/docs/evalci/overview)

</td>
</tr>
<tr>
<td width="50%" colspan="2">

**📊 Agent Benchmarking**<br/>
Evaluate your agents against industry-standard benchmarks like GAIA, SWE-bench, WebArena, and AgentBench directly from the CLI. Generate leaderboards to compare performance across tasks.

**🧪 EvalHub Community Suites**<br/>
Run shared community eval suites with `synapsekit bench` and compare aggregate score against baseline.
</td>
</tr>
</table>

</div>

### EvalHub quick usage

```bash
synapsekit bench --list
synapsekit bench --suite community/customer-support --model gpt-4o-mini
synapsekit bench --publish my_evals/ --name myorg/rag-finance
```

Docs: [docs/evalhub.md](docs/evalhub.md)

---

## Integrations

Everything plugs into the same interface. Swap any piece without rewriting your application logic.

### 🧠 LLM Providers

![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-CC785C?style=flat-square&logo=anthropic&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)
![AWS Bedrock](https://img.shields.io/badge/AWS%20Bedrock-FF9900?style=flat-square&logo=amazonaws&logoColor=white)
![Google Vertex AI](https://img.shields.io/badge/Vertex%20AI-4285F4?style=flat-square&logo=googlecloud&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral-FF7000?style=flat-square&logo=mistralai&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-39594D?style=flat-square&logo=cohere&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat-square&logo=groq&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF9D00?style=flat-square&logo=huggingface&logoColor=white)
![Cloudflare](https://img.shields.io/badge/Cloudflare-F48120?style=flat-square&logo=cloudflare&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white)
![Perplexity](https://img.shields.io/badge/Perplexity-20808D?style=flat-square&logo=perplexity&logoColor=white)
![Replicate](https://img.shields.io/badge/Replicate-000000?style=flat-square&logo=replicate&logoColor=white)
![xAI](https://img.shields.io/badge/xAI%20Grok-000000?style=flat-square&logo=x&logoColor=white)
![Baidu ERNIE](https://img.shields.io/badge/Baidu%20ERNIE-2932E1?style=flat-square&logo=baidu&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-4D6BFE?style=flat-square&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)
![Together AI](https://img.shields.io/badge/Together%20AI-000000?style=flat-square&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-6467F2?style=flat-square&logoColor=white)
![Fireworks AI](https://img.shields.io/badge/Fireworks%20AI-6D28D9?style=flat-square&logoColor=white)
![Cerebras](https://img.shields.io/badge/Cerebras-FF4B4B?style=flat-square&logoColor=white)
![SambaNova](https://img.shields.io/badge/SambaNova-E64A19?style=flat-square&logoColor=white)
![NovitaAI](https://img.shields.io/badge/NovitaAI-7C3AED?style=flat-square&logoColor=white)
![Writer](https://img.shields.io/badge/Writer-5B21B6?style=flat-square&logoColor=white)
![AI21 Labs](https://img.shields.io/badge/AI21%20Labs-4B5563?style=flat-square&logoColor=white)
![Aleph Alpha](https://img.shields.io/badge/Aleph%20Alpha-374151?style=flat-square&logoColor=white)
![Minimax](https://img.shields.io/badge/Minimax-1F2937?style=flat-square&logoColor=white)
![Moonshot](https://img.shields.io/badge/Moonshot%20Kimi-0F172A?style=flat-square&logoColor=white)
![Zhipu](https://img.shields.io/badge/Zhipu%20ChatGLM-1E3A5F?style=flat-square&logoColor=white)
![LM Studio](https://img.shields.io/badge/LM%20Studio-1A1A2E?style=flat-square&logoColor=white)
![llama.cpp](https://img.shields.io/badge/llama.cpp-4A4A4A?style=flat-square&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM-2D3748?style=flat-square&logoColor=white)
![GPT4All](https://img.shields.io/badge/GPT4All-3B4252?style=flat-square&logoColor=white)

---

### 🗄 Vector Stores

![In-Memory](https://img.shields.io/badge/In--Memory-6B7280?style=flat-square&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS%20(Meta)-0467DF?style=flat-square&logo=meta&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-24386C?style=flat-square&logo=qdrant&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=flat-square&logo=pinecone&logoColor=white)
![Weaviate](https://img.shields.io/badge/Weaviate-3DBE6C?style=flat-square&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-00A1EA?style=flat-square&logoColor=white)
![LanceDB](https://img.shields.io/badge/LanceDB-1E293B?style=flat-square&logoColor=white)
![PGVector](https://img.shields.io/badge/PGVector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![SQLiteVec](https://img.shields.io/badge/SQLiteVec-003B57?style=flat-square&logo=sqlite&logoColor=white)
![MongoDB Atlas](https://img.shields.io/badge/MongoDB%20Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)
![Redis](https://img.shields.io/badge/Redis%20Vector-DC382D?style=flat-square&logo=redis&logoColor=white)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-005571?style=flat-square&logo=elasticsearch&logoColor=white)
![OpenSearch](https://img.shields.io/badge/OpenSearch-003B57?style=flat-square&logo=opensearch&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase%20Vector-3ECF8E?style=flat-square&logo=supabase&logoColor=white)
![Cassandra](https://img.shields.io/badge/Cassandra-1287B1?style=flat-square&logo=apachecassandra&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)
![ClickHouse](https://img.shields.io/badge/ClickHouse-FFCC01?style=flat-square&logo=clickhouse&logoColor=black)
![Marqo](https://img.shields.io/badge/Marqo-6D28D9?style=flat-square&logoColor=white)
![Typesense](https://img.shields.io/badge/Typesense-D8014B?style=flat-square&logoColor=white)
![Vespa](https://img.shields.io/badge/Vespa-1A73E8?style=flat-square&logoColor=white)
![Zilliz](https://img.shields.io/badge/Zilliz-00A1EA?style=flat-square&logoColor=white)

---

### 📂 Data Loaders

**File Formats**

![PDF](https://img.shields.io/badge/PDF-FF0000?style=flat-square&logo=adobeacrobatreader&logoColor=white)
![Word](https://img.shields.io/badge/Word%20DOCX-2B579A?style=flat-square&logo=microsoftword&logoColor=white)
![Excel](https://img.shields.io/badge/Excel%20XLSX-217346?style=flat-square&logo=microsoftexcel&logoColor=white)
![PowerPoint](https://img.shields.io/badge/PowerPoint%20PPTX-B7472A?style=flat-square&logo=microsoftpowerpoint&logoColor=white)
![CSV](https://img.shields.io/badge/CSV-217346?style=flat-square&logoColor=white)
![JSON](https://img.shields.io/badge/JSON-292929?style=flat-square&logo=json&logoColor=white)
![Markdown](https://img.shields.io/badge/Markdown-000000?style=flat-square&logo=markdown&logoColor=white)
![HTML](https://img.shields.io/badge/HTML-E34F26?style=flat-square&logo=html5&logoColor=white)
![XML](https://img.shields.io/badge/XML-FF6600?style=flat-square&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?style=flat-square&logo=yaml&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=flat-square&logo=latex&logoColor=white)
![EPUB](https://img.shields.io/badge/EPUB-4A4A4A?style=flat-square&logoColor=white)
![RTF](https://img.shields.io/badge/RTF-4A4A4A?style=flat-square&logoColor=white)
![TSV](https://img.shields.io/badge/TSV-4A4A4A?style=flat-square&logoColor=white)
![Parquet](https://img.shields.io/badge/Parquet-50ABF1?style=flat-square&logo=apacheparquet&logoColor=white)
![Images](https://img.shields.io/badge/Images%20JPG%2FPNG-FF6B6B?style=flat-square&logoColor=white)
![Audio](https://img.shields.io/badge/Audio%20Whisper-FF9900?style=flat-square&logo=openai&logoColor=white)
![Video](https://img.shields.io/badge/Video%20MP4-FF0000?style=flat-square&logo=youtube&logoColor=white)

**Cloud Storage**

![AWS S3](https://img.shields.io/badge/AWS%20S3-569A31?style=flat-square&logo=amazons3&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=flat-square&logo=googledrive&logoColor=white)
![Azure Blob](https://img.shields.io/badge/Azure%20Blob-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)
![OneDrive](https://img.shields.io/badge/OneDrive-0078D4?style=flat-square&logo=microsoftonedrive&logoColor=white)
![Dropbox](https://img.shields.io/badge/Dropbox-0061FF?style=flat-square&logo=dropbox&logoColor=white)
![GCS](https://img.shields.io/badge/Google%20Cloud%20Storage-4285F4?style=flat-square&logo=googlecloud&logoColor=white)

**Databases**

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)
![DynamoDB](https://img.shields.io/badge/DynamoDB-FF9900?style=flat-square&logo=amazondynamodb&logoColor=white)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-005571?style=flat-square&logo=elasticsearch&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-4285F4?style=flat-square&logo=googlebigquery&logoColor=white)
![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=flat-square&logo=snowflake&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)

**APIs & Productivity**

![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)
![Jira](https://img.shields.io/badge/Jira-0052CC?style=flat-square&logo=jira&logoColor=white)
![Confluence](https://img.shields.io/badge/Confluence-172B4D?style=flat-square&logo=confluence&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white)
![Slack](https://img.shields.io/badge/Slack-4A154B?style=flat-square&logo=slack&logoColor=white)
![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat-square&logo=discord&logoColor=white)
![HubSpot](https://img.shields.io/badge/HubSpot-FF7A59?style=flat-square&logo=hubspot&logoColor=white)
![Salesforce](https://img.shields.io/badge/Salesforce-00A1E0?style=flat-square&logo=salesforce&logoColor=white)
![Airtable](https://img.shields.io/badge/Airtable-18BFFF?style=flat-square&logo=airtable&logoColor=white)
![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white)
![Reddit](https://img.shields.io/badge/Reddit-FF4500?style=flat-square&logo=reddit&logoColor=white)
![Wikipedia](https://img.shields.io/badge/Wikipedia-000000?style=flat-square&logo=wikipedia&logoColor=white)
![Google Sheets](https://img.shields.io/badge/Google%20Sheets-34A853?style=flat-square&logo=googlesheets&logoColor=white)
![Sitemap](https://img.shields.io/badge/Sitemap-FF6600?style=flat-square&logoColor=white)
![Obsidian](https://img.shields.io/badge/Obsidian-483699?style=flat-square&logo=obsidian&logoColor=white)
![arXiv](https://img.shields.io/badge/arXiv-B31B1B?style=flat-square&logoColor=white)
![PubMed](https://img.shields.io/badge/PubMed-326599?style=flat-square&logoColor=white)
![RSS](https://img.shields.io/badge/RSS-FFA500?style=flat-square&logo=rss&logoColor=white)
![Email](https://img.shields.io/badge/Email%20IMAP-EA4335?style=flat-square&logo=gmail&logoColor=white)
![Git Repo](https://img.shields.io/badge/Git%20Repo-F05032?style=flat-square&logo=git&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat-square&logo=firebase&logoColor=black)
![Twilio](https://img.shields.io/badge/Twilio-F22F46?style=flat-square&logo=twilio&logoColor=white)

---

### 🔧 Agent Tools

![DuckDuckGo](https://img.shields.io/badge/DuckDuckGo-DE5833?style=flat-square&logo=duckduckgo&logoColor=white)
![Google Search](https://img.shields.io/badge/Google%20Search-4285F4?style=flat-square&logo=google&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-000000?style=flat-square&logoColor=white)
![Wikipedia](https://img.shields.io/badge/Wikipedia-000000?style=flat-square&logo=wikipedia&logoColor=white)
![arXiv](https://img.shields.io/badge/arXiv-B31B1B?style=flat-square&logoColor=white)
![PubMed](https://img.shields.io/badge/PubMed-326599?style=flat-square&logoColor=white)
![YouTube](https://img.shields.io/badge/YouTube%20Search-FF0000?style=flat-square&logo=youtube&logoColor=white)
![Wolfram Alpha](https://img.shields.io/badge/Wolfram%20Alpha-DD1100?style=flat-square&logo=wolframalpha&logoColor=white)
![Slack](https://img.shields.io/badge/Slack-4A154B?style=flat-square&logo=slack&logoColor=white)
![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat-square&logo=discord&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub%20API-181717?style=flat-square&logo=github&logoColor=white)
![Jira](https://img.shields.io/badge/Jira-0052CC?style=flat-square&logo=jira&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white)
![Linear](https://img.shields.io/badge/Linear-5E6AD2?style=flat-square&logo=linear&logoColor=white)
![Stripe](https://img.shields.io/badge/Stripe-008CDD?style=flat-square&logo=stripe&logoColor=white)
![Twilio](https://img.shields.io/badge/Twilio-F22F46?style=flat-square&logo=twilio&logoColor=white)
![Google Calendar](https://img.shields.io/badge/Google%20Calendar-4285F4?style=flat-square&logo=googlecalendar&logoColor=white)
![AWS Lambda](https://img.shields.io/badge/AWS%20Lambda-FF9900?style=flat-square&logo=awslambda&logoColor=white)
![Playwright](https://img.shields.io/badge/Browser%20(Playwright)-2EAD33?style=flat-square&logo=playwright&logoColor=white)
![SQL](https://img.shields.io/badge/SQL%20Query-4479A1?style=flat-square&logo=mysql&logoColor=white)
![Python REPL](https://img.shields.io/badge/Python%20REPL-3776AB?style=flat-square&logo=python&logoColor=white)
![Shell](https://img.shields.io/badge/Shell%20Command-4EAA25?style=flat-square&logo=gnubash&logoColor=white)
![HTTP](https://img.shields.io/badge/HTTP%20Request-FF6600?style=flat-square&logoColor=white)

---

### 🧠 Memory Backends

![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![In-Memory](https://img.shields.io/badge/In--Memory-6B7280?style=flat-square&logoColor=white)

### 🗃 LLM Cache Backends

![Filesystem](https://img.shields.io/badge/Filesystem-4A4A4A?style=flat-square&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)
![Memcached](https://img.shields.io/badge/Memcached-009DC4?style=flat-square&logoColor=white)
![DynamoDB](https://img.shields.io/badge/DynamoDB-FF9900?style=flat-square&logo=amazondynamodb&logoColor=white)
![Semantic Cache](https://img.shields.io/badge/Semantic%20Cache-8E75B2?style=flat-square&logoColor=white)

### 📡 Observability

![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-425CC7?style=flat-square&logo=opentelemetry&logoColor=white)

---

## Install

**pip**
```bash
pip install synapsekit[openai]       # OpenAI
pip install synapsekit[anthropic]    # Anthropic
pip install synapsekit[ollama]       # Ollama (local)
pip install synapsekit[observe]      # Observability extras
pip install synapsekit[all]          # Everything
```

**uv**
```bash
uv add synapsekit[openai]
uv add synapsekit[all]
```

**Poetry**
```bash
poetry add synapsekit[openai]
poetry add "synapsekit[all]"
```

Full installation options → [docs](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/installation)

Observability guide → [docs/observability.md](docs/observability.md)

---

## Documentation

Everything you need to get started and go deep is in the docs.

| | |
|---|---|
| 🚀 [Quickstart](https://synapsekit.github.io/synapsekit-docs/docs/getting-started/quickstart) | Up and running in 5 minutes |
| 🗂 [RAG](https://synapsekit.github.io/synapsekit-docs/docs/rag/pipeline) | Pipelines, loaders, retrieval, vector stores |
| 🤖 [Agents](https://synapsekit.github.io/synapsekit-docs/docs/agents/overview) | ReAct, function calling, tools, executor |
| 🔀 [Graph Workflows](https://synapsekit.github.io/synapsekit-docs/docs/graph/overview) | DAG pipelines, conditional routing, parallel execution |
| 🧠 [LLM Providers](https://synapsekit.github.io/synapsekit-docs/docs/llms/overview) | All 34 providers with examples |
| 🧪 [EvalCI](https://synapsekit.github.io/synapsekit-docs/docs/evalci/overview) | LLM quality gates on every PR — GitHub Action |
| 📖 [API Reference](https://synapsekit.github.io/synapsekit-docs/docs/api/llm) | Full class and method reference |

---

## Development

```bash
git clone https://github.com/SynapseKit/SynapseKit
cd SynapseKit
uv sync --group dev
uv run pytest tests/ -q
```

---

## Contributing

Contributions are welcome — bug reports, documentation fixes, new providers, new features.

Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started. Look for issues tagged [`good first issue`](https://github.com/SynapseKit/SynapseKit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) if you're new.

---

## Community

- 💬 [Discord](https://discord.gg/PSuAXHRywJ) — chat, help, show and tell
- 💬 [Discussions](https://github.com/SynapseKit/SynapseKit/discussions) — ask questions, share ideas
- 🧭 [Discord roles draft](DISCORD_ROLES.md) — proposed roles and permissions for issue #389
- 🧭 [Discord release webhook draft](DISCORD_RELEASE_WEBHOOKS.md) — automate release announcements for issue #390
- 🐛 [Bug reports](https://github.com/SynapseKit/SynapseKit/issues/new?template=bug_report.yml)
- 💡 [Feature requests](https://github.com/SynapseKit/SynapseKit/issues/new?template=feature_request.yml)
- 🔒 [Security policy](SECURITY.md)

---

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AmitoVrito"><img src="https://avatars.githubusercontent.com/u/34062684?v=4" width="100px;" alt="Nautiverse"/><br /><sub><b>Nautiverse</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=AmitoVrito" title="Code">💻</a> <a href="https://github.com/SynapseKit/SynapseKit/commits?author=AmitoVrito" title="Documentation">📖</a> <a href="#maintenance-AmitoVrito" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gordienkoas"><img src="https://avatars.githubusercontent.com/u/127838071?v=4" width="100px;" alt="Gordienko Andrey"/><br /><sub><b>Gordienko Andrey</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=gordienkoas" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Deepak8858"><img src="https://avatars.githubusercontent.com/u/88921480?v=4" width="100px;" alt="Deepak singh"/><br /><sub><b>Deepak singh</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=Deepak8858" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/by22Jy"><img src="https://avatars.githubusercontent.com/u/122969909?v=4" width="100px;" alt="by22Jy"/><br /><sub><b>by22Jy</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=by22Jy" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Arjunkundapur"><img src="https://avatars.githubusercontent.com/u/64265396?v=4" width="100px;" alt="Arjun Kundapur"/><br /><sub><b>Arjun Kundapur</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=Arjunkundapur" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ashusf90"><img src="https://avatars.githubusercontent.com/u/153393197?v=4" width="100px;" alt="Harshit Gupta"/><br /><sub><b>Harshit Gupta</b></sub></a><br /><a href="https://github.com/SynapseKit/synapsekit-docs/pull/34" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DhruvGarg111"><img src="https://avatars.githubusercontent.com/u/136477030?v=4" width="100px;" alt="Dhruv Garg"/><br /><sub><b>Dhruv Garg</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=DhruvGarg111" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/adaumsilva"><img src="https://avatars.githubusercontent.com/u/178027480?v=4" width="100px;" alt="Adam Silva"/><br /><sub><b>Adam Silva</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=adaumsilva" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/qorexdev"><img src="https://avatars.githubusercontent.com/u/248982649?v=4" width="100px;" alt="qorex"/><br /><sub><b>qorex</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=qorexdev" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Abhay-Mmmm"><img src="https://avatars.githubusercontent.com/u/192120538?v=4" width="100px;" alt="Abhay Krishna"/><br /><sub><b>Abhay Krishna</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=Abhay-Mmmm" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ayushbhatt1224"><img src="https://avatars.githubusercontent.com/u/129763284?v=4" width="100px;" alt="AYUSH BHATT"/><br /><sub><b>AYUSH BHATT</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=ayushbhatt1224" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Chaturvediharsh123"><img src="https://avatars.githubusercontent.com/u/146837343?v=4" width="100px;" alt="HARSH"/><br /><sub><b>HARSH</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=Chaturvediharsh123" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mikemolinet"><img src="https://avatars.githubusercontent.com/u/237856306?v=4" width="100px;" alt="mikemolinet"/><br /><sub><b>mikemolinet</b></sub></a><br /><a href="https://github.com/SynapseKit/SynapseKit/commits?author=mikemolinet" title="Code">💻</a> <a href="https://github.com/SynapseKit/SynapseKit/issues?q=author%3Amikemolinet" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## License

[Apache 2.0](LICENSE)
