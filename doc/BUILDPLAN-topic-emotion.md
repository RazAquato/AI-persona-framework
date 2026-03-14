# Build Plan: Topic-Emotion Architecture

Phased implementation of per-persona fact scoping, topic salience tracking,
temporal pattern recognition, reflection agent, and persona autonomy.

Each phase is independently shippable and testable. Later phases build on
earlier ones but each phase delivers immediate value.

---

## Phase 1: Per-Persona Fact Scoping

**Goal:** Stop facts bleeding between personas. Core facts stay shared,
conversation-extracted facts stay with the persona they were told to.

### Step 1.1 — Schema migration
- Add `persona_id INT` column to `facts` table (nullable, FK to `user_personalities`)
- `NULL` = core fact (visible to all), non-NULL = persona-scoped
- Migration script: existing facts stay as `NULL` (core)
- Tests: verify column exists, NULL default works

### Step 1.2 — Fact store updates
- `store_fact()` and `store_fact_blobs()` accept optional `persona_id`
- `get_facts()` → `get_accessible_facts(user_id, persona_id)`:
  returns `WHERE persona_id IS NULL OR persona_id = ?`
- Tests: scoped facts only visible to owning persona, core facts visible to all

### Step 1.3 — Extraction scoping
- LLM knowledge extractor: emotional-tier facts get the current `persona_id`
- Identity-tier facts stay core (`persona_id = NULL`)
- External adapter facts (mealie, immich) always core
- Tests: extracted emotional facts tagged with persona_id

### Step 1.4 — Context builder update
- `context_builder.py` passes `persona_id` to fact retrieval
- Only surfaces core + current persona's scoped facts
- Tests: Dr. Lumen's emotional facts don't appear in Maya's context

### Step 1.5 — Prompt reframing
- Change fact injection phrasing in `prompt_builder.py` from
  "Here are facts about the user:" to
  "Background knowledge about the user. Only reference when directly
  relevant to the current conversation. Do not volunteer these unprompted:"
- Tests: prompt string contains new framing

---

## Phase 2: Topic Registry + Salience Tracking

**Goal:** Facts surface based on conversational relevance, not just existence.
Topics are first-class entities with per-persona salience that decays over time.

### Step 2.1 — Topic table
- Create `topics` table: `id, user_id, name, created_at`
- Create `topic_salience` table:
  `user_id, persona_id, topic_id, salience FLOAT, mention_count INT,
   last_mentioned TIMESTAMP, decay_floor FLOAT`
- `decay_floor` for sticky topics (e.g., family members never decay below 0.15)
- Tests: CRUD operations, unique constraints

### Step 2.2 — Tag-to-topic linking
- Facts already have `tags[]`. Create `fact_topics` junction table
  linking facts to topics via their tags.
- On fact insertion, auto-create topics for new tags and link them.
- Tests: new fact with tags creates topics and links

### Step 2.3 — Salience bumping
- When the engine processes a conversation turn, extract mentioned topics
  from the current message (reuse existing topic detection from knowledge extractor)
- Bump `salience` and `mention_count` for those topics with the current persona
- Formula: `salience = min(1.0, salience + 0.1 * (1 - salience))`
  (diminishing returns — converges toward 1.0)
- Tests: salience increases on mention, caps at 1.0

### Step 2.4 — Salience decay
- Add decay function: `salience *= decay_rate ^ days_since_last_mention`
- `decay_rate` ~0.97 (halves roughly every 23 days)
- Never drops below `decay_floor` (0.0 by default, 0.15 for person tags)
- Decay runs as part of nightly batch or on-access
- Tests: salience decays correctly, respects floor

### Step 2.5 — Salience-filtered context builder
- Context builder applies salience threshold (e.g., 0.2) when fetching facts
- Facts whose topics all have salience below threshold are excluded
- Facts with no topics always included (backward compatibility)
- Tests: low-salience facts excluded, high-salience included

---

## Phase 3: User Emotion Classification per Topic

**Goal:** The system understands HOW the user feels about each topic,
not just that the topic exists. Classified by nightly batch job.

### Step 3.1 — User-topic emotion table
- Create `user_topic_emotions` table:
  `user_id, topic_id, emotion STR, intensity FLOAT, last_updated TIMESTAMP`
- Uses the existing 18-emotion vocabulary (joy, anger, sadness, etc.)
- A topic can have multiple emotions simultaneously (love AND frustration)
- Tests: CRUD, multiple emotions per topic

### Step 3.2 — Batch emotion classifier
- New script: `memory-server/scripts/nightly_reflection.py`
- Reads today's chat messages grouped by session/persona
- For each topic discussed, classifies user's emotions from message context
- Uses the LLM with a structured prompt:
  "Given these messages about {topic}, classify the user's emotions.
   Return: [{emotion, intensity}]"
- Falls back to the existing regex `EmotionVectorGenerator` if LLM unavailable
- Tests: mock LLM response → correct emotion storage

### Step 3.3 — Emotion-aware context builder
- When injecting facts, include the user's emotional profile for related topics
- Prompt framing: "The user feels [frustration, nostalgia] about football.
  Handle this topic with care."
- Personas can then respond appropriately without being told explicitly
- Tests: emotional context appears in prompt for relevant topics

---

## Phase 4: Temporal Pattern Detection

**Goal:** The reflection agent notices trends and shifts in data over time.
Generates observations (not conclusions) as conversation hooks for personas.

### Step 4.1 — Observation table
- Create `observations` table:
  `id, user_id, topic_id, observation_text, observation_type
   (trend/shift/anomaly), confidence, created_at, surfaced BOOL`
- `surfaced` tracks whether a persona has used this observation yet
- Tests: CRUD operations

### Step 4.2 — Temporal analysis in reflection job
- Extend `nightly_reflection.py` to analyze temporal patterns:
  - Salience trends: "topic X declining over 3 months"
  - Adapter data shifts: "photo frequency with person Y dropped"
  - Emotion shifts: "user's feelings about Z changed from positive to mixed"
- Store as observations with confidence scores
- Tests: mock temporal data → correct observations generated

### Step 4.3 — Observation surfacing in prompts
- Context builder includes unsurfaced observations for current persona
  (filtered by salience threshold — only for topics the persona cares about)
- Prompt framing: "You've noticed a pattern: {observation}.
  You may explore this if it feels natural, but do not force it."
- Mark as surfaced after injection
- Tests: observations appear in prompt, get marked surfaced

---

## Phase 5: Persona Topic Emotions (Autonomy)

**Goal:** Personas develop their own feelings about topics, independent of
the user. Maya might grow bored of football. Dr. Lumen might develop concern.

### Step 5.1 — Persona-topic emotion table
- Create `persona_topic_emotions` table:
  `persona_id, user_id, topic_id, emotion STR, intensity FLOAT,
   last_updated TIMESTAMP`
- Separate from user-topic emotions — these are the persona's OWN feelings
- Tests: CRUD, independent from user emotions

### Step 5.2 — Personality tendency definitions
- Add `personality_tendencies` field to persona config (JSON):
  ```json
  {
    "bores_with_repetition": 0.7,
    "develops_genuine_interest": 0.5,
    "empathy_for_negative_topics": 0.8,
    "pushes_back_playfully": 0.6
  }
  ```
- These drive how the persona's emotions evolve
- Tests: tendency config loads correctly

### Step 5.3 — Persona emotion evolution in reflection job
- Extend `nightly_reflection.py`:
  - For each persona, look at topics discussed that day
  - Apply personality tendencies to generate persona emotional responses:
    - High `bores_with_repetition` + topic mentioned 15 times → annoyance grows
    - High `empathy_for_negative_topics` + user negative emotions → concern grows
    - High `develops_genuine_interest` + user positive emotions → interest grows
  - Store in persona_topic_emotions
- Tests: personality rules produce expected emotion evolution

### Step 5.4 — Persona emotions in prompt
- Include persona's own topic emotions in system prompt:
  "You feel [bored, slightly annoyed] about football because the user
  keeps bringing it up. You feel [genuine curiosity] about cooking."
- This makes the persona respond authentically, not as a mirror
- Tests: persona emotions appear in prompt, shape response framing

---

## Phase 6: Echo Integration

**Goal:** Feed the rich emotional profiles into Echo's corpus builder
for a deeper, more authentic digital legacy.

### Step 6.1 — Echo emotional profile builder
- Extend `corpus_builder.py` to include user-topic emotion profiles
- Only identity-tier topics + positive/neutral valence emotions
- Output: "Kenneth was passionate about technology (fascination, pride),
  loved his children deeply (love, pride, tenderness), found cooking
  meditative (contentment, satisfaction)"
- Tests: emotional profiles included in corpus, negative emotions filtered

### Step 6.2 — Echo temporal narrative
- Include temporal patterns in Echo's knowledge:
  "Kenneth went through a period of intense interest in X during 2023-2024"
- Observations that have been confirmed by the user get promoted to
  identity-tier narratives
- Tests: temporal narratives appear in Echo corpus

---

## Implementation Notes

- **Each phase is independently deployable** — Phase 1 alone solves the
  immediate oversharing problem
- **Nightly reflection job** (Phase 3+) runs as cron, same pattern as
  sync adapters: `scripts/nightly_reflection.py`
- **No breaking changes** — new tables, new columns are nullable, existing
  facts continue working
- **Test counts** — each step adds unit tests. Target: maintain 100%
  of existing tests passing + new coverage for each step
- **LLM dependency** — Phases 3-5 use the LLM for classification, with
  regex/heuristic fallbacks when the LLM is offline
- **Database impact** — new tables are lightweight. Topic salience is the
  most-written table (updated per conversation turn) — consider batch
  updates if performance matters

---

## Estimated Scope

| Phase | Steps | Core Change |
|-------|-------|-------------|
| 1 | 5 | Per-persona facts + prompt reframing |
| 2 | 5 | Topic registry + salience + decay |
| 3 | 3 | User emotion classification per topic |
| 4 | 3 | Temporal pattern detection + observations |
| 5 | 4 | Persona autonomy + personality tendencies |
| 6 | 2 | Echo enrichment |

Phase 1-2 fix the immediate UX problems. Phase 3-5 build the deep
personality system. Phase 6 completes the Echo vision.
