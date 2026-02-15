from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date

import pandas as pd

from src.common.config import AppConfig, get_config
from src.data.clients import NsePriceClient, YahooFinanceClient
from src.data.contracts import IngestionRequest, SymbolIngestionResult, UniverseIngestionResult
from src.data.filters import UniverseFilterEngine
from src.data.quality import add_turnover, normalize_ohlcv, validate_price_frame

logger = logging.getLogger(__name__)


class MarketDataIngestionService:
    def __init__(
        self,
        config: AppConfig | None = None,
        nse_client: NsePriceClient | None = None,
        yahoo_client: YahooFinanceClient | None = None,
    ) -> None:
        self.config = config or get_config()
        self.nse_client = nse_client or NsePriceClient()
        self.yahoo_client = yahoo_client or YahooFinanceClient()
        self.filter_engine = UniverseFilterEngine(self.config.universe)

    def fetch_symbol_data(self, symbol: str, start_date: date, end_date: date) -> SymbolIngestionResult:
        nse_prices = self.nse_client.fetch_ohlcv(symbol, start_date, end_date)

        if nse_prices.empty:
            logger.warning("NSEpy returned no price data; using Yahoo fallback", extra={"symbol": symbol})
            prices = self.yahoo_client.fetch_ohlcv(symbol, start_date, end_date)
        else:
            prices = nse_prices

        prices = normalize_ohlcv(prices)
        prices = add_turnover(prices)

        ok, errors = validate_price_frame(prices)
        if not ok:
            logger.warning("Price data validation failed", extra={"symbol": symbol, "errors": errors})

        fundamentals = self.yahoo_client.fetch_fundamentals(symbol)
        return SymbolIngestionResult(symbol=symbol.upper(), ohlcv=prices, fundamentals=fundamentals)

    def fetch_universe_data(self, request: IngestionRequest) -> UniverseIngestionResult:
        price_frames: list[pd.DataFrame] = []
        fundamentals_records: list[dict] = []

        for symbol in request.symbols:
            try:
                result = self.fetch_symbol_data(symbol=symbol, start_date=request.start_date, end_date=request.end_date)
                if not result.ohlcv.empty:
                    price_frames.append(result.ohlcv)
                fundamentals_records.append(result.fundamentals)
            except Exception as exc:  # pragma: no cover
                logger.exception("Symbol ingestion failed", extra={"symbol": symbol, "error": str(exc)})

        all_prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
        all_fundamentals = pd.DataFrame(fundamentals_records)

        filter_report = self.filter_engine.build_filter_report(all_prices, all_fundamentals)
        eligible_symbols = set(filter_report.loc[filter_report["eligible"], "symbol"].tolist())

        eligible_prices = all_prices[all_prices["symbol"].isin(eligible_symbols)].reset_index(drop=True)
        eligible_fundamentals = all_fundamentals[all_fundamentals["symbol"].isin(eligible_symbols)].reset_index(drop=True)

        logger.info(
            "Universe ingestion complete",
            extra={
                "request": asdict(request),
                "total_symbols": len(request.symbols),
                "eligible_symbols": len(eligible_symbols),
            },
        )

        return UniverseIngestionResult(
            ohlcv=eligible_prices,
            fundamentals=eligible_fundamentals,
            filter_report=filter_report,
        )
