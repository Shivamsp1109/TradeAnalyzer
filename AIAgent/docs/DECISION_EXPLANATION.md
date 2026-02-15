# Decision and Explanation Layer (Step 6)

## Implemented Logic
- Decision rule checks:
  - expected 60-day return > configured minimum
  - probability(return > 8%) > configured minimum
  - volatility below configured threshold
  - fundamentals not deteriorating
  - model confidence above fail-safe floor
- Entry logic:
  - default current price
  - if near MA20 within tolerance, use MA20 pullback entry
- Target logic:
  - `target_price = entry_price * (1 + selected_expected_return)`
- Horizon selection:
  - compares 30/60/90-day risk-adjusted expected returns
  - selects highest score
- Explanation:
  - template-based summary using momentum, fundamentals, model output, and selected horizon
- Disclaimers injected in response:
  - Educational purposes only
  - Not investment advice
  - Markets involve risk

## Modules
- `src/decision/engine.py`
- `src/decision/contracts.py`
- `src/explanation/templates.py`
- `src/api/recommendation_service.py`

## Integration point
`RecommendationBuilderService.build(inference, latest_row)` returns a `RecommendationResponse` payload ready for FastAPI output.
