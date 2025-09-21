# FantasyManagerv2

A Streamlit workbench for blending sportsbook player prop lines, FantasyPros projections and
league data from `nfl_data_py`. The initial milestone focuses on the projection explorer tab that
lets you refresh projections for a date window and compare players across books (treating
FantasyPros as another "book").

## Project layout

```
app/
  config.py             # environment loading helpers
  data_manager.py       # orchestrates pulls + local caching
  data_sources/         # integrations for Odds API, FantasyPros, nfl_data_py
  projections.py        # fantasy scoring + aggregation utilities
  utils.py              # shared helpers (naming, filesystem)
data/
  sample_data/          # lightweight starter projections for local exploration
streamlit_app.py        # UI entrypoint (v0 projection workbench)
```

## Getting started

1. **Install dependencies** (requires Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure secrets**:

   ```bash
   cp .env.example .env
   # Edit .env and paste your The Odds API key
   ```

3. **Run the app**:

   ```bash
   streamlit run streamlit_app.py
   ```

   The default window is 21–25 September 2025. Use the sidebar to adjust the window or trigger an
   update (limited to once per hour unless you tick "force").

4. **Data storage**: fetched artefacts are written to `data/processed/` as Parquet files. When the
   cache is empty the app falls back to `data/sample_data/*.json` so you always have something to
   explore without hitting the APIs.

## Development notes

- API calls live in `app/data_sources/`. Each module isolates HTTP usage to simplify future testing
  or mocking.
- `app/projections.py` mirrors the scoring logic provided in the brief and exposes helpers for
  computing PPR, half-PPR, DraftKings and CBS points per source.
- `DataManager` enforces a one-hour update interval (persisted in `data/state.json`).
- The UI currently ships with the projection workbench; future tabs (lineup assistant, accuracy
  tracker, advanced stats explorer) can plug into the same data layer.

## Testing

Run the unit tests with:

```bash
pytest
```

> Note: the optional dependencies (`pandas`, `streamlit`, etc.) are required for the full feature
> set. The tests cover the pieces that do not hit external services.
