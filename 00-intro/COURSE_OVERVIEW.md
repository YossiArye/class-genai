# Welcome: What Is GenAI, Why Do You Need It, and What Will You Build?

## 0. AI, ML, GenAI, LLM: who is who?

Each term sits inside the previous one: **AI ⊃ Machine Learning ⊃ Deep Learning ⊃ GenAI ⊃ LLM**.

| Term | What it means | Think of it as |
|---|---|---|
| **AI** | Any machine acting intelligently, even by hand-written rules | Following the driving rulebook: "red light, stop" |
| **Machine Learning** | Learning from examples instead of written rules | Learning to drive with an instructor who corrects every move |
| **Deep Learning** | ML with many-layered neural networks, powerful enough for language and images | A veteran driver's automatic intuition |
| **GenAI** | Models that *create* new content instead of only classifying or deciding | A veteran driver who can write a brand-new driving lesson |
| **LLM** | The GenAI engine for text, trained to predict the next word | A child who has heard millions of sentences and completes "Once upon a…" |

Some models also learn with no teacher at all, by trial, reward, and penalty, like a baby learning
to walk (*reinforcement learning*). You won't train models in this course. You will **build
products on top of LLMs**, the last two rows.

## 1. What is Generative AI?

Traditional software follows rules that a developer wrote by hand: `if` this, then that.
Traditional machine learning *classifies or predicts* from data: spam or not spam, price next month.

**Generative AI (GenAI)** *creates* new content: text, code, images, audio, video, structured data.
It does this with models, mainly **Large Language Models (LLMs)**, trained on enormous amounts of
data to predict what comes next.

The key ideas, which Module 00 makes concrete with a few lines of numpy:

- **Everything becomes numbers.** Text is split into *tokens*, and each token becomes a vector (an *embedding*).
- **Meaning becomes geometry.** Similar meanings sit close together in vector space. This is why
  "King − Man + Woman ≈ Queen", and it is the basis of semantic search.
- **A model is a very large weighted sum.** It multiplies, adds, and repeats billions of times to
  score the most likely next token, and that single step, repeated, produces whole answers.

You don't need a math background. You need the intuition, and you will get it by running code.

## 2. Why do we need it?

Because an LLM is the first technology that lets software **understand human language and
unstructured data**, which is most of the world's data (documents, emails, chats, PDFs, images).

| Before GenAI | With GenAI |
|---|---|
| Search matches keywords | Search understands meaning and intent |
| Users learn your UI, filters, and query language | Users just ask in plain language |
| Extracting data from documents needs custom parsers per format | One model reads any format and returns clean JSON |
| Automation breaks on anything outside the rules | Agents decide the next step and handle the unexpected |
| Reports and SQL need an analyst | "How many orders did we have last quarter?" gets an answer |
| Support teams answer the same questions repeatedly | An assistant answers from your own documentation |

But a raw LLM alone is not enough for real products. It:

- **doesn't know your data**: your documents, your database, your customers;
- **hallucinates**: it can state wrong things confidently;
- **can't act**: by itself it only produces text;
- **forgets**: it has no memory between calls;
- **is unpredictable**: free text is hard to plug into code.

**This course is about closing those gaps.** Each module fixes one of them:

| The problem | The solution you will learn |
|---|---|
| Doesn't know your data | **RAG** and vector databases (Modules 02, 04), **GraphRAG** (11) |
| Hallucinates | Grounding in retrieved sources, reranking, and **evaluation metrics** (04) |
| Can't act | **Function calling and tools** (07), **MCP** (10) |
| Can't plan or remember | **Agents with LangGraph**: state, memory, loops (08) |
| Unpredictable output | **Structured output** and validation (01) |
| Can't read data in plain language | **Text-to-SQL** and SQL agents (06, 09) |
| Text only | **Multimodal**: images, image generation, video (05) |

## 3. What you will be able to do at the end

After the course you will be able to **design, build, and evaluate GenAI applications**, not just
call an API. Concretely, you will be able to:

- **Explain** how LLMs work (tokens, embeddings, next-token prediction) and debug their behavior.
- **Build a chatbot over your own documents** (RAG): chunk, embed, store in ChromaDB or FAISS, retrieve, and answer with sources.
- **Improve and measure it**: hybrid search, reranking, and metrics that show whether a change really helped.
- **Make models return reliable output**: structured JSON your code can trust.
- **Compose pipelines** with LangChain and LCEL, and swap OpenAI for a local Ollama model without rewriting your app.
- **Give an LLM tools**: connect it to APIs, databases, and your own functions.
- **Build autonomous agents** with LangGraph that plan, use tools, keep memory, and coordinate with other agents.
- **Let anyone query a database in natural language**, with safe, validated SQL.
- **Work with images and video** alongside text.
- **Build and consume MCP servers**, so your tools work with any MCP-compatible client.
- **Use GraphRAG** for questions that need connecting facts across many documents.
- **Choose the right approach**: cloud or local model, RAG or agent, and what it costs.

You will leave with a portfolio of working, documented projects, from a document Q&A bot to a
tool-using agent to a SQL assistant with memory. That is the skill set behind the "AI engineer" role.
