# Build Plan: Topic-Emotion Architecture

Phased implementation of domain-based knowledge access, session grouping,
topic salience tracking, temporal pattern recognition, reflection agent,
and persona autonomy.

Each phase is independently shippable and testable. Later phases build on
earlier ones but each phase delivers immediate value.

---

## Phase 1: Knowledge Domains + Persona Access Control

**Goal:** Facts are categorized into knowledge domains. Each persona only
sees domains it has access to. Replaces the current "everyone sees everything"
model with a permission system that mirrors how real relationships work.

### Domains

| Domain | Examples |
|--------|----------|
| Family | Children, wife, parents, relatives, family events |
| Physical | Health, fitness, workouts, body, medical |
| Hobbies | Football, cooking, photography, gaming, music |
| Work | Job, career, skills, colleagues, projects |
| Emotional | Feelings, moods, mental health, struggles |
| Memories | Life events, travel, nostalgia, milestones |
| Other | Uncategorized / general knowledge |

### Default persona access

| Persona | Domains |
|---------|---------|
| Dr. Lumen | Family, Physical, Hobbies, Work, Emotional, Memories (all) |
| Maya | Physical, Hobbies, Work |
| Coach | Physical, Hobbies |
| DebugBot | Work |

Access is user-configurable per persona.

### Step 1.1 — Schema: domains + access
- Create `knowledge_domains` table: `id, name, description`
- Seed default domains (Family, Physical, Hobbies, Work, Emotional, Memories, Other)
- Add `domain TEXT` column to `facts` table (nullable — NULL treated as "Other")
- Add `domain_access TEXT[]` column to `user_personalities` table (or separate junction table)
- Migration: existing facts get `domain = NULL` (visible via "Other")
- Tests: tables exist, defaults seeded, column nullable

### Step 1.2 — Fact store domain support
- `store_fact()` and `store_fact_blobs()` accept optional `domain` parameter
- New: `get_accessible_facts(user_id, persona_id)` — fetches facts where
  `domain IN (persona's domain_access)` OR `domain IS NULL`
- Existing `get_facts()` unchanged for backward compat (returns all)
- Tests: domain-filtered facts respect access list

### Step 1.3 — Extraction domain classification
- LLM knowledge extractor: add `domain` to the JSON extraction schema
  (enum of domain names — grammar-constrained like tier already is)
- Regex extractor fallback: classify by keyword heuristics
  (family names → Family, health terms → Physical, etc.)
- External adapters set domain explicitly:
  - Mealie → Hobbies
  - Immich people → Family
  - Immich locations → Memories
  - Immich devices → Other
- Tests: extracted facts get correct domains

### Step 1.4 — Context builder domain filtering
- `context_builder.py` loads persona's `domain_access` list
- Passes to `get_accessible_facts()` — only surfaces permitted domains
- Tests: Dr. Lumen sees Family facts, Maya does not

### Step 1.5 — Conversation-private facts
- Add `persona_id INT` column to `facts` table (nullable)
- Facts extracted from conversation with emotional tier → scoped to that persona
  (regardless of domain)
- Context builder: include facts where `persona_id IS NULL OR persona_id = current`
- This layer sits ON TOP of domain access — private facts skip the domain check
- Tests: emotional fact told to Dr. Lumen not visible to Maya even if same domain

### Step 1.6 — Prompt reframing
- Change fact injection in `prompt_builder.py`:
  Old: "Here are facts about the user:"
  New: "Background knowledge about the user. Only reference when directly
  relevant to the current conversation. Do not volunteer these unprompted."
- Tests: prompt string contains new framing

### Step 1.7 — Persona domain config UI
- API endpoint: `PUT /personas/{id}/domains` — update domain access list
- Web UI: checkboxes in persona settings for which domains to enable
- New persona creation: starts with empty domain access (user must configure)
- Tests: API updates domain_access, UI renders checkboxes

---

## Phase 1B: Session Groups (Project Folders)

**Goal:** Users can organize chat sessions into named groups, like project
folders in other chatbots. Groups are per-persona. Provides structure that
later phases (topic salience, reflection) can leverage.

### Step 1B.1 — Session groups table
- Create `session_groups` table:
  `id SERIAL, user_id INT, persona_id INT, name TEXT, created_at TIMESTAMP`
- Add `group_id INT` column to `chat_sessions` table (nullable FK)
- Ungrouped sessions continue working (NULL group_id)
- Tests: CRUD operations, FK constraint

### Step 1B.2 — API endpoints
- `GET /session-groups` — list groups for user, optionally filtered by persona
- `POST /session-groups` — create group (name + persona_id)
- `PUT /session-groups/{id}` — rename group
- `DELETE /session-groups/{id}` — delete group (sessions become ungrouped, not deleted)
- `PUT /sessions/{id}/group` — move session into a group (or NULL to ungroup)
- Tests: CRUD + ownership checks

### Step 1B.3 — UI: folder view in sidebar
- Session list grouped by folders (collapsible)
- Ungrouped sessions shown at bottom
- "New folder" button
- Drag-and-drop or context menu to move sessions into folders
- Folder rename/delete via context menu
- Tests: UI renders folder structure

### Step 1B.4 — Session creation with group
- When creating a new chat, optionally select a group
- "New Chat" button uses current group if one is selected
- Tests: new session respects group_id

---

## Phase 2: Topic Registry + Salience Tracking

**Goal:** Topics are first-class entities ("football", "painting", "Trym")
with per-persona salience that rises with conversation and decays over time.
Controls which facts surface in prompts based on conversational relevance.

### Step 2.1 — Topic table
- Create `topics` table: `id, user_id, name, created_at`
- Create `topic_salience` table:
  `user_id INT, persona_id INT, topic_id INT, salience FLOAT,
   mention_count INT, last_mentioned TIMESTAMP, decay_floor FLOAT`
- `decay_floor` for sticky topics (family members: 0.15, general: 0.0)
- Unique constraint: (user_id, persona_id, topic_id)
- Tests: CRUD operations, unique constraints

### Step 2.2 — Tag-to-topic linking
- Facts already have `tags[]`. Create `fact_topics` junction table
  linking facts to topics via their tags.
- On fact insertion, auto-create topics for new tags and link them
- Topics are normalized lowercase, deduplicated
- Tests: new fact with tags creates topics and links

### Step 2.3 — Salience bumping (real-time)
- When the engine processes a conversation turn, extract mentioned topics
  (reuse topic detection from knowledge extractor)
- Bump salience for those topics with the current persona:
  `salience = min(1.0, salience + 0.1 * (1 - salience))`
  (diminishing returns — converges toward 1.0)
- Increment `mention_count`, update `last_mentioned`
- Tests: salience increases on mention, caps at 1.0

### Step 2.4 — Salience decay (nightly)
- Decay function: `salience *= decay_rate ^ days_since_last_mention`
- `decay_rate` ≈ 0.97 (halves roughly every 23 days)
- Never drops below `decay_floor`
- Runs as part of nightly batch job
- Tests: salience decays correctly, respects floor, old topics fade

### Step 2.5 — Salience-filtered context builder
- Context builder applies salience threshold (e.g., 0.2) when building prompt
- For each fact, check if ANY of its linked topics have salience above threshold
  for the current persona
- Facts with no linked topics: always included (backward compat)
- Order facts by max topic salience (most relevant first)
- Tests: low-salience facts excluded, high-salience included, ordering correct

### Step 2.6 — Session group salience aggregation
- Topics mentioned across sessions in the same group get a group-level
  salience boost (the project folder is inherently about those topics)
- When resuming a session in a group, the group's topics get a small
  context boost even if not mentioned in THIS specific session
- Tests: group membership boosts topic salience

---

## Phase 3: User Emotion Classification per Topic

**Goal:** The system understands HOW the user feels about each topic,
not just that the topic exists. Two dimensions: salience (how much you
care) and emotional profile (how you feel). Classified by nightly batch job.

### Step 3.1 — User-topic emotion table
- Create `user_topic_emotions` table:
  `user_id INT, topic_id INT, emotion TEXT, intensity FLOAT,
   last_updated TIMESTAMP`
- Uses the existing 18-emotion vocabulary
- A topic can have multiple emotions simultaneously (love AND frustration)
- Tests: CRUD, multiple emotions per topic, upsert on re-classification

### Step 3.2 — Nightly emotion classifier
- New script: `memory-server/scripts/nightly_reflection.py`
- Reads today's chat messages grouped by session/persona
- For each topic discussed, classifies user's emotions from message context
- Uses the LLM with a structured prompt + grammar-constrained output:
  "Given these messages about {topic}, what emotions does the user express?
   Return: [{emotion, intensity}]"
- Falls back to existing regex `EmotionVectorGenerator` if LLM unavailable
- Emotions accumulate over time (running average, not replaced daily)
- Tests: mock LLM response → correct emotion storage

### Step 3.3 — Emotion-aware context builder
- When injecting facts, include the user's emotional profile for related topics
- Prompt framing:
  High-salience + negative emotions: "The user has strong negative feelings
  about {topic} (frustration, resentment). Do not bring this up unless the
  user does. If they do, respond with empathy."
  High-salience + positive emotions: "The user enjoys {topic}. Safe to
  reference naturally."
  High-salience + mixed: "The user has complex feelings about {topic}.
  Approach with nuance."
- Tests: emotional context appears in prompt, framing matches emotion valence

---

## Phase 4: Temporal Pattern Detection

**Goal:** The reflection agent notices trends and shifts in data over time.
Generates observations (not conclusions) as conversation hooks for personas.
Data is not insight — the system observes, the user explains.

### Step 4.1 — Observation table
- Create `observations` table:
  `id SERIAL, user_id INT, topic_id INT, observation_text TEXT,
   observation_type TEXT (trend/shift/anomaly), confidence FLOAT,
   created_at TIMESTAMP, surfaced_to JSONB DEFAULT '{}'`
- `surfaced_to` tracks which personas have seen this observation:
  `{"persona_id_7": "2026-04-01", ...}`
- Tests: CRUD operations

### Step 4.2 — Temporal analysis in reflection job
- Extend `nightly_reflection.py` to analyze temporal patterns:
  - Salience trends: "topic X has been declining over 3 months"
  - Adapter data shifts: "photo frequency with Trym dropped significantly
    in 2025 compared to 2023"
  - Emotion shifts: "user's feelings about {topic} changed from positive
    to frustrated over the past 2 weeks"
  - Workout/health trends (future: Fitbit adapter)
- Store as observations with confidence scores
- Observations are factual statements about DATA, never interpretations
- Tests: mock temporal data → correct observations generated

### Step 4.3 — Observation surfacing in prompts
- Context builder includes unsurfaced observations for current persona
  (filtered by: persona has domain access to the observation's topic domain,
  AND topic salience > threshold for this persona)
- Prompt framing: "You've noticed a pattern: {observation}.
  You may gently explore this if the conversation goes there naturally.
  Do not force it. Do not assume you know the reason — ask."
- Mark as surfaced for this persona after injection
- Tests: observations appear in prompt, get marked surfaced per persona

---

## Phase 5: Persona Topic Emotions (Autonomy)

**Goal:** Personas develop their own feelings about topics, independent of
the user. Maya might grow bored of football. Dr. Lumen might develop concern.
Personas are characters, not mirrors.

### Step 5.1 — Persona-topic emotion table
- Create `persona_topic_emotions` table:
  `persona_id INT, user_id INT, topic_id INT, emotion TEXT,
   intensity FLOAT, last_updated TIMESTAMP`
- Completely separate from user-topic emotions
- Tests: CRUD, independent from user emotions

### Step 5.2 — Personality tendency definitions
- Add `personality_tendencies JSONB` column to `user_personalities` table
- Default tendencies per persona:
  ```
  Maya (girlfriend):
    bores_with_repetition: 0.7      # gets annoyed if topic is overused
    develops_genuine_interest: 0.6   # picks up enthusiasm from user
    empathy_for_negative_topics: 0.5 # moderate sympathy
    pushes_back_playfully: 0.8      # teases, has opinions

  Dr. Lumen (psychiatrist):
    bores_with_repetition: 0.1      # never bored, it's clinical data
    develops_genuine_interest: 0.3   # professional distance
    empathy_for_negative_topics: 0.9 # deep concern
    pushes_back_playfully: 0.1      # doesn't tease

  Coach (trainer):
    bores_with_repetition: 0.4
    develops_genuine_interest: 0.7   # gets excited about your goals
    empathy_for_negative_topics: 0.4
    pushes_back_playfully: 0.6      # motivational tough love

  DebugBot (debug):
    bores_with_repetition: 0.0      # doesn't care, it's a tool
    develops_genuine_interest: 0.0
    empathy_for_negative_topics: 0.0
    pushes_back_playfully: 0.0
  ```
- Users can customize these for custom personas
- Tests: tendencies load from DB, defaults applied on seed

### Step 5.3 — Persona emotion evolution in reflection job
- Extend `nightly_reflection.py`:
  - For each persona, examine topics discussed that day
  - Apply personality tendencies to evolve persona emotions:
    - High `bores_with_repetition` + topic mentioned many times recently
      → annoyance/boredom grows
    - High `empathy_for_negative_topics` + user has negative emotions
      → concern/compassion grows
    - High `develops_genuine_interest` + user has positive emotions
      → interest/enthusiasm grows
    - High `pushes_back_playfully` + any strong emotion
      → persona develops its own contrarian take
  - Emotions decay toward neutral when topic is inactive
  - Store in persona_topic_emotions
- Tests: personality rules produce expected emotion evolution

### Step 5.4 — Persona emotions in prompt
- Include persona's own topic emotions in system prompt:
  "Your feelings about topics the user has discussed:
  - Football: you feel bored and slightly annoyed (the user keeps ranting)
  - Cooking: you feel genuinely curious (you've grown to enjoy hearing about it)
  - Trym: you feel warmth (you can tell the user loves talking about his son)
  Let these feelings influence your tone naturally. You are not obligated
  to agree with or mirror the user."
- Tests: persona emotions appear in prompt, shape response framing

---

## Phase 6: Echo Integration

**Goal:** Feed the rich emotional profiles into Echo's corpus builder
for a deeper, more authentic digital legacy.

### Step 6.1 — Echo emotional profile builder
- Extend `corpus_builder.py` to include user-topic emotion profiles
- Only identity-tier topics + positive/neutral valence emotions
- Only topics with sufficient history (salience ever exceeded 0.5)
- Output: "Kenneth was passionate about technology (fascination, pride),
  loved his children deeply (love, pride, tenderness), found cooking
  meditative (contentment, satisfaction)"
- Tests: emotional profiles included, negative-only topics filtered

### Step 6.2 — Echo temporal narrative
- Include temporal patterns in Echo's knowledge:
  "Kenneth went through a period of intense interest in home automation
  during 2025-2026"
- Observations confirmed by user conversation get promoted to
  identity-tier narratives
- Tests: temporal narratives appear in Echo corpus

---

## Implementation Notes

- **Each phase is independently deployable.** Phase 1 alone fixes the
  immediate oversharing problem Kenneth observed in beta testing.
- **No breaking changes.** New tables, new nullable columns. Existing
  facts, sessions, and personas continue working throughout.
- **Nightly reflection job** (Phase 3+) runs as cron alongside existing
  sync adapters. Pattern: `scripts/nightly_reflection.py`
- **LLM dependency.** Phase 1 needs no LLM changes. Phase 2 reuses existing
  topic detection. Phases 3-5 use the LLM for classification with
  regex/heuristic fallbacks when offline.
- **Test discipline.** Each step adds unit tests. All existing 433 tests
  must continue passing. Target: 500+ tests by Phase 2 completion.
- **UI changes.** Phase 1 (domain checkboxes), Phase 1B (session folders),
  later phases add topic/emotion displays as needed.

---

## Estimated Scope

| Phase | Steps | Core Change |
|-------|-------|-------------|
| 1     | 7     | Knowledge domains + persona access + prompt reframing |
| 1B    | 4     | Session groups (project folders) |
| 2     | 6     | Topic registry + salience tracking + decay |
| 3     | 3     | User emotion classification per topic |
| 4     | 3     | Temporal pattern detection + observations |
| 5     | 4     | Persona autonomy + personality tendencies |
| 6     | 2     | Echo enrichment |

**Phase 1 + 1B** fix the immediate UX problems (oversharing, organization).
**Phase 2** adds intelligence to what surfaces in conversation.
**Phase 3-5** build the deep personality system.
**Phase 6** completes the Echo vision.
