# Data Ingestion (Step 3)

## Purpose
The ingestion layer pulls EOD price/fundamental data for Indian equities and filters the universe before feature generation.

## Modules
- `src/data/clients.py`
  - `NsePriceClient`: primary OHLCV source (NSEpy)
  - `YahooFinanceClient`: fallback OHLCV + fundamentals
- `src/data/quality.py`
  - OHLCV normalization and validation
  - turnover derivation (`close * volume`)
- `src/data/filters.py`
  - Universe filters for:
    - min close price
    - min median daily turnover
    - min market cap
  - rejection reasons appended in `reasons`
- `src/data/ingestion.py`
  - Orchestrates per-symbol ingestion and returns eligible universe
- `src/data/contracts.py`
  - Typed request/response data structures

## Typical Usage
```python
from datetime import date
from src.data import IngestionRequest, MarketDataIngestionService

svc = MarketDataIngestionService()
result = svc.fetch_universe_data(
    IngestionRequest(
        symbols=["RELIANCE", "TCS", "INFY"],
        start_date=date(2022, 1, 1),
        end_date=date(2026, 1, 31),
    )
)

# result.ohlcv -> only eligible symbols
# result.fundamentals -> only eligible symbols
# result.filter_report -> includes ineligible reasons
```
