# Batch 6 Summary — External Adapters + Model Switching

## What Was Done

### 1. Mealie Recipe Sync (shared/tools/mealie_sync.py)
Syncs meal plan history from Mealie into identity-tier facts. Uses meal plan entries (not just recipes) to count how often each dish is cooked — frequency reveals real preferences. "Taco cooked 5 times" gets higher confidence than "Fish cooked once".

**Facts produced:** recipe frequency, food categories, frequent ingredients (2+ recipes).
**Real sync for user 52:** 5 meal plan entries → 5 recipes → 8 facts stored.

### 2. Immich Photo Archive Sync (shared/tools/immich_sync.py)
Syncs photo metadata from Immich into identity-tier facts. Extracts three categories:
- **People** — named faces ranked by photo count (Ylva 956, Trym 890, Kenneth 352...)
- **Locations** — cities from EXIF GPS data, sampled across years (Lysaker, Billingstad, Kjenn...)
- **Devices** — cameras used (iPhone 13 mini, Sony J9110)

**Real sync for user 52:** 20 people, 5 cities, 1 country, 3 devices → 29 facts stored.

### 3. Cron Script (archive/cron_sync_adapters.sh)
Monthly cron script that runs both Mealie and Immich sync for user 52. Ready to install:
```
0 3 1 * * /home/kenneth/AI-persona-framework/archive/cron_sync_adapters.sh >> .../sync.log 2>&1
```

### 4. Model Switching (LLM-client/core/model_manager.py + UI)
Full model hot-swap from the web UI:
- **Backend:** `model_manager.py` — kills running llama-server (SIGTERM → SIGKILL fallback), waits for port, launches new instance, polls /health until ready (60s timeout)
- **API:** `GET /models` lists available models with VRAM sizes; `POST /model/switch` triggers the swap
- **UI:** Dropdown in chat header shows all models with VRAM size. Selecting a different model shows a "Switch" button. Clicking it opens a confirmation modal ("service will be down for a minute"). After confirming, the backend handles the transition.
- **model_configs.yaml:** Updated with human-readable `name` and `vram_gb` for each model

### 5. User Setup
- User 52 renamed from `SeedUser` to `kenneth`, password set to `test123`
- 37 new facts stored for user 52 (8 mealie + 29 immich)
- All personas will now see these facts in conversation context

## Files Created/Modified

| File | Action |
|------|--------|
| `shared/tools/immich_sync.py` | Created — Immich sync adapter |
| `shared/tools/mealie_sync.py` | Already existed from earlier work |
| `shared/tests/test_immich_sync.py` | Created — 14 tests |
| `memory-server/scripts/sync_immich.py` | Created — CLI entry point |
| `archive/cron_sync_adapters.sh` | Created — monthly cron script |
| `LLM-client/core/model_manager.py` | Created — process management for llama-server |
| `LLM-client/tests/test_model_manager.py` | Created — 9 tests |
| `LLM-client/tests/test_api.py` | Modified — added 3 model endpoint tests |
| `LLM-client/interface/api/app.py` | Modified — model dropdown UI, switch modal, API endpoints |
| `LLM-client/config/model_configs.yaml` | Modified — added name, vram_gb fields |

## Test Results

**433 tests passing** across all three modules (87 LLM-client + 150 memory-server + 196 shared).

## What's Next

- Install cron job for monthly syncs (`crontab -e`)
- Reflection job — synthesize emotional-tier facts into identity-tier narratives
- Home automation integration via Neo4j house graph
