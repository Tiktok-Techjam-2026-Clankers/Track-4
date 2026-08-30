# Track 4: Shopping Copilot — AI Conversational Search and Recommendations

## 4.1 Background

Traditional e-commerce search engines heavily rely on static keyword matching, failing to capture the fluid shifts of genuine consumer psychology and the distinction between open-ended browsing and high-intent buying. In modern conversational commerce, constructing an intelligent agent that leverages dynamic context programming is critical to bridging the gap between ambiguous user queries and complex product catalogs. Solving this challenge directly impacts core industrial metrics.

## 4.2 Problem Statement

Participants are challenged to architect an intelligent, next-generation shopping agent capable of navigating real-world customer dynamics. Moving beyond rigid search filters, the engineered system must demonstrate deep cognitive understanding, runtime architectural agility, and commercial efficiency using the provided Amazon dataset.

The system should be built upon the following four core pillars:

### I. Core Architecture: Intent Routing & Hybrid Pipeline

- **Dual-Track Routing:** Instantly detect the user's underlying intent — triggering a high-precision filter track for targeted "Buying" to lock hard constraints, and a diverse dense retrieval track for open-ended "Browsing" to unlock cross-category scenario matching.
- **Pipeline Base:** Construct an in-memory data stream featuring "Multi-Route Retrieval → LLM Semantic Ranking" (combining keyword, category, and vector similarity).

### II. Dialog Strategy: Multi-Turn Scenario Evolution

- **Dynamic State Machine:** Build a robust conversational state tracker to gracefully handle dynamic Information Accumulation (incremental slots) and abrupt Intent Override (slot erasure and rewriting).
- **Proactive Guidance:** Trigger an immediate retrieval cutoff when facing Over-Generality (candidate pool overload) to actively generate structured, proactive clarification prompts that guide user convergence.

### III. Self-Evolution: Dynamic Context Programming

- **Runtime Adaptation:** Leverage accumulated dialog history to perform Personalized Context Distillation, continuously updating short-term session states and long-term user profiles.
- **Adaptive Orchestration:** Utilize dynamic Context Programming to achieve runtime workflow re-orchestration and strategy alignment, ensuring the agent iteratively refines its own guidance logic.

### IV. Evaluation Matrix: Product & Efficiency Metrics

Anchored on the final purchased record within the Amazon dataset, performance is quantified across three dimensions:

- **Coverage (Hit Rate@K):** Measures the catalog recall and boundary capability during the retrieval stage.
- **Precision (MRR / Top-K Hit Rate):** Evaluates the LLM's accuracy in pushing the exact purchased item to the absolute top of the recommendation list.
- **Efficiency (MTTC — Mean Turns to Conversion):** Heavy rewards systems that guide the user to the correct product in fewer interaction rounds, penalizing unnecessary conversational cognitive load.

## 4.3 Constraints & Scope

| Category | Details |
|---|---|
| **In scope** | Designing highly sensitive intent-detection modules to split traffic into "Buying" and "Browsing" tracks. Implementing heterogeneous retrieval routing (weights, custom dynamic truncation, and slot decay over time). Engineering runtime-adaptive memory layers for personalized context distillation. Fine-tuning prompt strategies or local scoring logic for the LLM ranking stage to compress decision paths. |
| **Out of scope** | UI/UX Development (evaluated purely via automated backend APIs and headless pipelines). Training or full-parameter fine-tuning of base foundational LLMs. Deploying heavy external industrial vector DB clusters (must run entirely in-memory for light execution). Multi-Modal Processing (restricted strictly to text catalogs, structured metadata, and text dialogs). |
| **Limits** | Max Turns: Hard limit of 10 turns per session (forced termination and zero score if exceeded). Catalog Mutation: The Amazon product dataset is strictly read-only; no structural mutations or mock ASIN injections are allowed. |
| **Allowed assumptions** | Inputs are pre-cleaned text strings (no spelling correction, typos, or ASR noise). Product catalog, pricing, and category trees are static for the duration of the hackathon. Each session is simulated as an isolated single-user interaction (no multi-user concurrency). |

## 4.4 Available Resources & Data

### Competition Data

- A frozen catalog containing 50,000 products from the Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` category.
- 200 labeled public development sessions for local testing and iteration.
- 800 additional sessions retained privately by the organizer for final evaluation.
- Public and private evaluation sessions use separate users and target products.

Both splits use the same fixed scenario mix:

- 40% Buying — a hard constraint is disclosed early
- 40% Browsing — the customer begins vague
- 15% Intent Override — an earlier preference is replaced on turn 3 or 4
- 5% Boundary — the customer may have no preference for a requested attribute

### Participant Resources

- A weak BM25 starter Agent implemented in Python.
- A deterministic local evaluator for Hit Rate@10, MRR, MTTC, Efficiency, and TechnicalScore.
- A published Python Agent interface and machine-readable API contract.
- Evaluation configuration, reproducible baseline results, data documentation, and submission rules.
- A SHA256 checksum file for verifying the downloaded catalog.

The organizer does not provide hosted model access, API keys, model tokens, or third-party API credits. A paid LLM is not required to complete the challenge.

### Required Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

- `message`: customer-facing natural language
- `ask_attribute`: one allowed attribute or `null`
- `recommendations`: ordered best to worst, max 10 valid unique IDs scored
- `usage`: non-negative prompt and completion token counts (optional when no model used)

### Metrics

```text
HitRate@10     = successful sessions / N
MRR            = sum(1 / target_rank, misses = 0) / N
MTTC           = sum(first_hit_turn, misses = 11) / N
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
```

| Setting | Value |
|---|---|
| Catalog ID field | `parent_asin` |
| Top K | 10 |
| Max turns | 10 |
| Miss turn value | 11 |
| Exact match | true |
| Scenario metrics | buying, browsing, intent_override, boundary |

### Baseline

Weak BM25 baseline on 200 public sessions:

| Metric | Value |
|---|---|
| Hit Rate@10 | 0.125 |
| MRR | 0.068 |
| MTTC | 9.81 |
| TechnicalScore | 0.107 |

### Resources

- Participant repository: https://github.com/TechJam2026/techjam-conversational-search
- Participant Kit Release: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
- Original data source: https://amazon-reviews-2023.github.io/

## 4.5 Deliverables

1. **Written Project Description (via Devpost)**
   - How your solution addresses the problem statement
   - Development tools used (e.g. VSCode, Colab, Jupyter)
   - APIs used (e.g. OpenAI GPT-4o, Google Maps API)
   - Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)
   - Datasets and assets used

2. **Public Code/GitHub Repository**
   - Well-structured, commented code covering all components
   - A README with: project overview, setup instructions, steps to reproduce results, limitations and improvements, team member contributions

3. **Demo Video**
   - Demonstrates solution working end-to-end
   - Uploaded to YouTube (public visibility)
   - Linked in Devpost description
   - For backend/NLP tracks: a walkthrough showing API usage, inference examples, or result analysis is accepted

## 4.6 Judging Criteria

| Criteria | Definition | Weight |
|---|---|---:|
| **Technical Execution** | Strong engineering fundamentals — well-structured code, thoughtful architecture, effective use of APIs or models. The demo runs reliably, and the technical complexity reflects deliberate, capable decision-making. | 35% |
| **Innovation & Problem Insight** | Originality in both idea and approach. Stands out for sharpness of problem understanding — how clearly the team has framed the challenge, why it matters, and how directly the solution addresses it. | 20% |
| **Impact & Relevance** | Clear potential to deliver value to real users or stakeholders — meaningful reach, tangible benefit, and relevance beyond the hackathon prompt. | 20% |
| **Feasibility & Practicality** | Realistic and buildable beyond a prototype. Technically and operationally sustainable — resource usage is proportionate, architecture holds under real-world conditions. | 15% |
| **Presentation & Communication** | [Final Event Only] The team communicates with clarity. The pitch tells a coherent story from problem to solution to potential, and the team responds to questions with depth. | 10% |

## Model Policy

Teams choose and manage their own model credentials. API keys must be passed through environment variables and never committed. For official final scoring, the organizer may disable network access — submissions must document whether they require network access and describe any offline fallback.
