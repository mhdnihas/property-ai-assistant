# Architecture and System Design

This document explains the Property AI Assistant at three levels, each with diagrams:

1. [Business view](#1-business-view): who uses it, what problem it solves, how it creates value
2. [Technical architecture](#2-technical-architecture): the main building blocks and how data moves
3. [Detailed system design](#3-detailed-system-design): components, request flow, algorithms,
   failure handling, data model and deployment

The diagrams are written in [Mermaid](https://mermaid.js.org/), which GitHub renders
automatically. To edit one, paste it into https://mermaid.live.

---

## 1. Business view

### 1.1 Business context

Property buyers and investors have to search two very different sources: a developer's
marketing site (DarGlobal: international branded residences) and a property portal (Wasalt:
thousands of individual Saudi listings). The assistant puts both behind one conversation,
answers in plain language and links back to the original pages. It sends people to the
sources rather than replacing them.

```mermaid
flowchart LR
    subgraph People["Who asks"]
        BUYER["Home buyer<br/>looking in Saudi cities"]
        INVESTOR["International investor<br/>branded residences, residency"]
        AGENT["Broker / sales agent<br/>quick facts for clients"]
    end

    ASSISTANT(["Property AI Assistant<br/>one chat, cited answers"])

    subgraph Sources["Public sources of truth"]
        DG["DarGlobal website<br/>projects, amenities, delivery dates,<br/>investment and visa info, FAQ"]
        WS["Wasalt portal<br/>listings with price, district, size,<br/>FAQs and market guides"]
    end

    subgraph Outcomes["Business outcomes"]
        LEAD["Qualified interest<br/>visit project page, book unit, enquire"]
        DISCOVERY["Faster discovery<br/>filtered shortlist in seconds"]
        TRUST["Trust<br/>every fact links to its source"]
    end

    BUYER --> ASSISTANT
    INVESTOR --> ASSISTANT
    AGENT --> ASSISTANT
    DG -. "scraped, indexed" .-> ASSISTANT
    WS -. "scraped, indexed" .-> ASSISTANT
    ASSISTANT --> LEAD
    ASSISTANT --> DISCOVERY
    ASSISTANT --> TRUST
    LEAD -. "click-through" .-> DG
    LEAD -. "click-through" .-> WS
```

### 1.2 Use cases

```mermaid
flowchart TB
    USER(("User"))

    subgraph DGUC["DarGlobal (international developer)"]
        UC1["Learn about a project<br/>'Tell me about Trump Tower Jeddah'"]
        UC2["Browse projects by country<br/>'Which projects are in Oman?'"]
        UC3["Investment and residency<br/>'How do I get Saudi residency?'"]
    end

    subgraph WSUC["Wasalt (Saudi property portal)"]
        UC4["Search listings with criteria<br/>'Villas in Riyadh under 2M'"]
        UC5["Refine by follow-up<br/>'what about Jeddah?'"]
        UC6["Market insight<br/>'Average villa price in Riyadh?'"]
    end

    subgraph Guard["Guard rails"]
        UC7["Out-of-scope question<br/>answers honestly: no information"]
        UC8["Manipulation attempt<br/>refuses, stays on topic"]
    end

    USER --> UC1 & UC2 & UC3 & UC4 & UC6 & UC7 & UC8
    UC4 --> UC5
```

### 1.3 User journey

```mermaid
journey
    title Investor looking for a villa in Saudi Arabia
    section Discover
      Open the assistant link: 5: User
      Pick a suggested question: 4: User
    section Search
      Ask for villas in Riyadh under 2M SAR: 5: User, Assistant
      Read a table of matching listings: 5: User
      Ask what about Jeddah: 5: User, Assistant
    section Evaluate
      Ask about residency through investment: 4: User, Assistant
      Check the cited sources: 4: User
    section Act
      Open the listing on Wasalt: 5: User
      Enquire with DarGlobal: 4: User
```

### 1.4 Value and constraints

| Goal | How the design supports it |
|---|---|
| Answers people can trust | Retrieval-augmented: the model only sees retrieved documents, and every fact carries an `[n]` link to its source page |
| Respect budgets and criteria | Price, city, type and bedrooms are applied as strict filters in code, never left to the model |
| Zero running cost | Free OpenRouter models, a free CPU embedding model, free Render hosting |
| Always available for reviewers | The data is scraped offline and shipped inside the image, so the demo doesn't depend on the source sites being up |
| Responsible data use | Only public pages, robots.txt respected, polite rate, bot protection not bypassed |

---

## 2. Technical architecture

### 2.1 High-level architecture

The system has two halves. The **offline data plane** runs on a developer machine and turns
the websites into a search index committed to git. The **online serving plane** is a single
container that answers questions from that index.

```mermaid
flowchart LR
    subgraph EXT1["Source websites"]
        DGW["darglobal.co.uk"]
        WSW["wasalt.sa"]
        WSC["cdn.wasalt.sa<br/>public sitemaps"]
    end

    subgraph OFF["Offline data plane: developer machine"]
        direction TB
        SCR["Scrapers<br/>browser extractors + sitemap script"]
        RAW[("data/raw<br/>JSON / JSONL")]
        DOCS["build_documents<br/>clean, dedupe, chunk, normalise"]
        PROC[("data/processed<br/>documents.jsonl")]
        IDX["build_index<br/>embed, atomic write"]
        INDEX[("data/index<br/>docs + vectors + manifest")]
        SCR --> RAW --> DOCS --> PROC --> IDX --> INDEX
    end

    GIT[("Git repository")]

    subgraph ON["Online serving plane: Docker container on Render"]
        direction TB
        UI["Web UI<br/>static HTML/JS"]
        API["FastAPI<br/>/api/chat, /api/health"]
        RAG["RAG core<br/>parse, retrieve, prompt"]
        LLMC["LLM client<br/>stream + fallback"]
        MEM[("In-memory index<br/>BM25 + vectors")]
        UI <--> API --> RAG --> LLMC
        RAG --> MEM
    end

    OR["OpenRouter<br/>free LLMs"]
    USER(("User browser"))

    DGW --> SCR
    WSW --> SCR
    WSC --> SCR
    INDEX --> GIT --> |"git push, Docker build"| ON
    USER <--> |"HTTPS, SSE stream"| UI
    LLMC <--> |"HTTPS, SSE stream"| OR
```

### 2.2 Technology choices

```mermaid
flowchart TB
    subgraph Frontend
        F1["Vanilla JS + CSS<br/>no build step"]
        F2["fetch + ReadableStream<br/>SSE parsing"]
        F3["Safe Markdown renderer<br/>escape first, then format"]
    end
    subgraph Backend
        B1["FastAPI + Uvicorn<br/>async, OpenAPI docs"]
        B2["pydantic / pydantic-settings<br/>validation, config"]
        B3["httpx<br/>async streaming HTTP"]
    end
    subgraph Retrieval
        R1["rank-bm25<br/>keyword ranking"]
        R2["fastembed + ONNX<br/>bge-small-en-v1.5, CPU"]
        R3["NumPy<br/>cosine similarity, RRF"]
    end
    subgraph Generation
        G1["OpenRouter API<br/>OpenAI-compatible"]
        G2["Free models<br/>Nemotron, Gemma"]
    end
    subgraph Ops
        O1["Docker multi-stage<br/>non-root, health check"]
        O2["Render<br/>hosting, secrets, auto-deploy"]
        O3["pytest + ruff<br/>tests, lint, format"]
    end
    Frontend --> Backend --> Retrieval
    Backend --> Generation
    Ops -.-> Backend
```

---

## 3. Detailed system design

### 3.1 Component diagram

Each box is a module in the repository. Arrows mean "calls" or "depends on".

```mermaid
flowchart TB
    subgraph static["app/static"]
        HTML["index.html"]
        APPJS["app.js<br/>chat state, SSE reader"]
        MDJS["markdown.js<br/>safe renderer, citations"]
    end

    subgraph main["app/main.py"]
        FACTORY["create_app()<br/>lifespan wiring"]
        MW["middleware<br/>request ID, security headers, access log"]
        RL["RateLimiter<br/>sliding window per IP"]
        CHAT["POST /api/chat<br/>StreamingResponse"]
        HEALTH["GET /api/health"]
    end

    subgraph rag["app/rag"]
        SVC["service.py<br/>ChatService"]
        QP["query_parser.py<br/>parse_query, QueryFilters"]
        VOC["vocab.py<br/>city / type aliases"]
        RET["retriever.py<br/>HybridRetriever"]
        EMB["embeddings.py<br/>Embedder"]
        PR["prompt.py<br/>SYSTEM_PROMPT, build_user_prompt"]
        MOD["models.py<br/>Document, ScoredDocument"]
    end

    subgraph llm["app/llm"]
        ORC["openrouter.py<br/>OpenRouterClient"]
    end

    CFG["app/config.py<br/>Settings"]
    IDXF[("data/index")]
    ORAPI["OpenRouter API"]

    APPJS --> CHAT
    APPJS --> HEALTH
    APPJS --> MDJS
    FACTORY --> CFG
    FACTORY --> SVC
    CHAT --> RL
    CHAT --> SVC
    SVC --> QP --> VOC
    SVC --> RET
    SVC --> PR
    SVC --> ORC
    RET --> EMB
    RET --> MOD
    RET --> IDXF
    ORC --> ORAPI
```

### 3.2 Request sequence: `POST /api/chat`

The full path of one question, including the streaming protocol and the model fallback.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Browser UI (app.js)
    participant API as FastAPI (main.py)
    participant S as ChatService
    participant P as QueryParser
    participant R as HybridRetriever
    participant L as OpenRouterClient
    participant OR as OpenRouter

    U->>UI: types "Villas in Riyadh under 2M"
    UI->>API: POST /api/chat {message, history}
    API->>API: validate schema, length limit
    API->>API: rate limit check (per IP)
    alt over the limit
        API-->>UI: 429 + Retry-After
    end
    API->>S: stream(message, history)
    S->>P: parse_query(message)
    P-->>S: QueryFilters(city, purpose, type, max_price)
    opt short follow-up (8 words or fewer)
        S->>P: parse_query(previous question)
        S->>S: merge missing filters from previous question
    end
    S->>R: search(query, filters, k=8)
    R->>R: drop listings that fail filters
    R->>R: BM25 rank + embedding rank, RRF fusion
    R-->>S: 8 documents + matching count
    S-->>API: event "sources"
    API-->>UI: SSE sources
    S->>S: build system prompt + numbered context
    S->>L: stream_chat(messages)
    loop models in order (cooling models last)
        L->>OR: POST /chat/completions (stream, reasoning off)
        alt 429 / 5xx / timeout before first token
            OR-->>L: error
            L->>L: cool down model if 429/503, try next
        else success
            OR-->>L: SSE deltas
        end
    end
    loop each delta
        L-->>S: Delta(text, model)
        S-->>API: event "token"
        API-->>UI: SSE token
        UI->>UI: render Markdown, link [n] citations
    end
    S-->>API: event "done" {model, elapsed}
    API-->>UI: SSE done
    UI->>U: answer + source cards
```

### 3.3 Retrieval algorithm

```mermaid
flowchart TD
    Q["User question"] --> PARSE["parse_query()<br/>city, purpose, type,<br/>bedrooms, price range"]
    PARSE --> FU{"Short follow-up<br/>and previous question?"}
    FU -- yes --> MERGE["Prepend previous question to query<br/>inherit missing filters"]
    FU -- no --> MASK
    MERGE --> MASK["Build allowed-mask<br/>listing docs must satisfy every filter<br/>non-listing docs always allowed"]
    MASK --> BM["BM25 scores<br/>top 50 allowed, score > 0"]
    MASK --> DENSE["Embedding cosine scores<br/>top 50 allowed"]
    BM --> RRF["Reciprocal Rank Fusion<br/>score = sum of 1 / (60 + rank)"]
    DENSE --> RRF
    RRF --> BOOST{"Listing intent<br/>and filters present?"}
    BOOST -- yes --> B2["Boost matching listings<br/>+1/60 each"]
    BOOST -- no --> SEL1["Take top k = 8"]
    B2 --> SEL2["Up to 6 listings<br/>+ remaining slots for context<br/>FAQ, market guides"]
    SEL1 --> OUT["RetrievalResult<br/>documents, filters, matching count"]
    SEL2 --> OUT
```

Why each part exists:

| Step | Reason |
|---|---|
| Filters before ranking | "Under 2 million" must be a guarantee, not a suggestion to the model |
| BM25 | Exact names and numbers ("Pagani", "W Residences", listing IDs) |
| Embeddings | Meaning and paraphrases ("place to live near the sea" ↔ "seafront residences") |
| RRF | Combines two rankings with different score scales, no tuning needed |
| Listing / context split | Users get both concrete listings and background (price norms, FAQs) |

### 3.4 LLM client: fallback and cool-down

```mermaid
stateDiagram-v2
    [*] --> OrderModels: stream_chat(messages)
    OrderModels: Order models<br/>ready first, cooling last
    OrderModels --> TryModel

    TryModel: POST to next model<br/>stream, reasoning disabled
    TryModel --> Streaming: HTTP 200 and first token
    TryModel --> AuthError: HTTP 401 / 403
    TryModel --> Retryable: 429, 5xx, 4xx model error,<br/>timeout, network error, empty answer

    Retryable --> CoolDown: status 429 or 503
    Retryable --> NextModel: other errors
    CoolDown: Skip this model for 60 s
    CoolDown --> NextModel
    NextModel --> TryModel: models left
    NextModel --> AllBusy: no models left

    Streaming --> Streaming: delta to user
    Streaming --> Done: stream finished
    Streaming --> Interrupted: error after first token

    AuthError --> [*]: LLMError, API key rejected
    AllBusy --> [*]: LLMError, all AI models busy
    Interrupted --> [*]: LLMError, answer interrupted
    Done --> [*]
```

Design rule: **never switch models after the first token**. An answer stitched together from
two models would be incoherent, so a mid-stream failure is reported to the user instead.

### 3.5 Offline data pipeline

```mermaid
flowchart TD
    subgraph S1["1. Collect"]
        A1["darglobal_extract.js<br/>65 pages, 1 every 1.5 s"]
        A2["wasalt_extract.js<br/>32 category pages, 1 every 2 s"]
        A3["wasalt_sitemap.py<br/>60,000 listing URLs"]
    end
    subgraph S2["2. Build documents: build_documents.py"]
        B1["DarGlobal pages<br/>drop errors / duplicates,<br/>strip boilerplate, chunk 1,400 chars"]
        B2["Project directory<br/>one document per country"]
        B3["Wasalt listings<br/>normalise city/type, price/sqm,<br/>structured meta"]
        B4["FAQs + market guides<br/>dedupe, chunk"]
        B5["Sitemap statistics<br/>one summary document"]
    end
    subgraph S3["3. Build index: build_index.py"]
        C1{"Same id + text<br/>in previous index?"}
        C2["Reuse vector"]
        C3["Embed with bge-small"]
        C4["Write to temp dir"]
        C5["Atomic directory swap"]
    end
    A1 --> B1 & B2
    A2 --> B3 & B4
    A3 --> B5
    B1 & B2 & B3 & B4 & B5 --> D[("1,484 documents")]
    D --> C1
    C1 -- yes --> C2 --> C4
    C1 -- no --> C3 --> C4
    C4 --> C5 --> IDX[("data/index<br/>committed to git")]
```

### 3.6 Data model

```mermaid
classDiagram
    class Document {
        +str id
        +Source source
        +Kind kind
        +str title
        +str url
        +str text
        +dict meta
        +embedding_text() str
    }
    class ListingMeta {
        +int listing_id
        +str city
        +str district
        +str purpose
        +str property_type
        +int bedrooms
        +float price
        +float land_area_sqm
    }
    class ScoredDocument {
        +Document document
        +float score
    }
    class QueryFilters {
        +str city
        +str purpose
        +str property_type
        +int bedrooms
        +int min_bedrooms
        +float min_price
        +float max_price
        +bool listing_intent
        +has_constraints() bool
        +matches(meta) bool
    }
    class RetrievalResult {
        +list~ScoredDocument~ documents
        +QueryFilters filters
        +int matching_listings
    }
    class ChatEvent {
        +str type
        +dict data
    }
    class Delta {
        +str text
        +str model
    }
    Document "1" --> "0..1" ListingMeta : meta when kind = listing
    ScoredDocument --> Document
    RetrievalResult --> "*" ScoredDocument
    RetrievalResult --> QueryFilters
    QueryFilters ..> ListingMeta : matches()
```

Allowed values:

| Field | Values |
|---|---|
| `Document.source` | `darglobal`, `wasalt` |
| `Document.kind` | `project`, `page`, `catalogue`, `listing`, `faq`, `market_guide`, `market_stats` |
| `ListingMeta.purpose` | `sale`, `rent` |
| `ListingMeta.property_type` | `Villa`, `Apartment`, `Land`, `Building`, `Floor` (normalised by `vocab.py`) |
| `ChatEvent.type` | `sources`, `token`, `done`, `error` |

### 3.7 Streaming protocol (Server-Sent Events)

```mermaid
sequenceDiagram
    participant API as Server
    participant UI as Browser
    API->>UI: event sources, data {"sources": [...8 items]}
    Note over UI: keeps sources to link [n] citations
    API->>UI: event token, data {"t": "Here are "}
    API->>UI: event token, data {"t": "villas in Riyadh [1]..."}
    Note over UI: re-renders Markdown once per animation frame
    alt success
        API->>UI: event done, data {"model": "...", "elapsed_s": 4.2}
    else failure
        API->>UI: event error, data {"message": "user-safe text"}
    end
    Note over UI: shows source cards, stores turn in history
```

### 3.8 Deployment

```mermaid
flowchart LR
    subgraph DEV["Developer machine"]
        CODE["Code + data/index"]
        ENV[".env<br/>local key, git-ignored"]
        TEST["make check<br/>61 unit tests + lint"]
        CODE --> TEST
    end

    GH[("GitHub<br/>mhdnihas/property-ai-assistant")]

    subgraph RENDER["Render: free web service, Singapore"]
        BUILD["Docker build<br/>multi-stage, ~450 MB"]
        SECRET["Environment variable<br/>OPENROUTER_API_KEY, encrypted"]
        subgraph CT["Container: non-root uid 1000, PORT=10000"]
            UV["Uvicorn + FastAPI"]
            IDXC[("data/index<br/>+ embedding model baked in")]
        end
        HC["Health check<br/>GET /api/health"]
        PROXY["HTTPS edge (Cloudflare)<br/>property-ai-assistant.onrender.com"]
    end

    USERS(("Users"))
    ORX["OpenRouter"]

    TEST -->|"git push origin main"| GH -->|"auto-deploy on commit"| BUILD --> CT
    SECRET -. "env var at runtime" .-> UV
    USERS -->|HTTPS| PROXY -->|"X-Forwarded-For"| UV
    UV -->|"HTTPS, Bearer key"| ORX
    HC -.-> UV
```

### 3.9 Failure modes and how they are handled

```mermaid
flowchart LR
    F1["Free model rate-limited"] --> H1["Fall back to next model,<br/>cool down 60 s"]
    F2["All models busy"] --> H2["SSE error event,<br/>friendly message"]
    F3["Bad / missing API key"] --> H3["Fail fast, no retries;<br/>health shows llm_configured=false"]
    F4["Index files out of sync"] --> H4["Log error, serve with<br/>BM25-only search"]
    F5["Interrupted index build"] --> H5["Atomic swap keeps<br/>previous index intact"]
    F6["Embedding model fails to load"] --> H6["BM25-only retrieval"]
    F7["Too many requests from one IP"] --> H7["429 + Retry-After"]
    F8["Oversized / invalid input"] --> H8["422 before any work"]
    F9["Prompt injection in question"] --> H9["System prompt: context is data,<br/>stay on topic"]
    F10["Model returns HTML / script"] --> H10["UI escapes before rendering"]
```

### 3.10 Scaling path

The current design targets a demo with one replica. If traffic grows, the changes would be:

| Concern | Today | At scale |
|---|---|---|
| Rate limiting | In-memory per process | Redis or the API gateway |
| Index | Loaded in memory (~2 MB of vectors) | Vector database (pgvector, Qdrant) once there are millions of listings |
| Freshness | Manual re-scrape | Scheduled job: scrape, build, test, deploy |
| LLM cost / limits | Free models, ~50 requests/day per free key | Paid key or self-hosted model, response cache for repeated questions |
| Observability | Structured logs with request IDs | Metrics (latency, model used, fallback rate) and tracing |
| Quality | Smoke test with 7 questions | Evaluation set with expected sources, run in CI |
