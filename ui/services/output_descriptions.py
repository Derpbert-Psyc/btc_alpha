"""Human-readable descriptions for every indicator output."""

OUTPUT_DESCRIPTIONS = {
    # EMA (ID 1)
    ("ema", "ema"): "Smoothed price value",

    # RSI (ID 2)
    ("rsi", "rsi"): "Relative strength (0-100, >70 overbought, <30 oversold)",

    # ATR (ID 3)
    ("atr", "atr"): "Average true range (volatility in price units)",

    # Pivot Structure (ID 4)
    ("pivot_structure", "pivot_high"): "Most recent swing high price",
    ("pivot_structure", "pivot_low"): "Most recent swing low price",

    # AVWAP (ID 5)
    ("avwap", "avwap"): "Anchored volume-weighted average price",

    # DD Equity (ID 6)
    ("dd_equity", "equity_dd"): "Current equity drawdown from peak (%)",
    ("dd_equity", "equity_dd_duration"): "Bars since equity peak",

    # MACD (ID 7) — also macd_tv
    ("macd", "macd_line"): "Fast EMA minus slow EMA (momentum)",
    ("macd", "signal_line"): "Smoothed MACD line (signal crossover reference)",
    ("macd", "histogram"): "MACD line minus signal line (acceleration)",
    ("macd", "slope_sign"): "MACD line direction: +1 rising, -1 falling, 0 flat",
    ("macd", "signal_slope_sign"): "Signal line direction: +1 rising, -1 falling, 0 flat",
    ("macd_tv", "macd_line"): "Fast EMA minus slow EMA (momentum)",
    ("macd_tv", "signal_line"): "Smoothed MACD line (signal crossover reference)",
    ("macd_tv", "histogram"): "MACD line minus signal line (acceleration)",
    ("macd_tv", "slope_sign"): "MACD line direction: +1 rising, -1 falling, 0 flat",
    ("macd_tv", "signal_slope_sign"): "Signal line direction: +1 rising, -1 falling, 0 flat",

    # ROC (ID 8)
    ("roc", "roc"): "Rate of change (% price change over period)",

    # ADX (ID 9)
    ("adx", "adx"): "Trend strength (0-100, >25 trending, <20 ranging)",
    ("adx", "plus_di"): "Positive directional indicator (bullish pressure)",
    ("adx", "minus_di"): "Negative directional indicator (bearish pressure)",

    # Choppiness (ID 10)
    ("choppiness", "choppiness"): "Choppiness index (0-100, >61 choppy, <38 trending)",

    # Bollinger (ID 11)
    ("bollinger", "basis"): "Middle band (SMA of price)",
    ("bollinger", "upper"): "Upper band (basis + N standard deviations)",
    ("bollinger", "lower"): "Lower band (basis - N standard deviations)",
    ("bollinger", "bandwidth"): "Band width as % of basis (volatility measure)",
    ("bollinger", "percent_b"): "Price position within bands (0=lower, 1=upper)",

    # LinReg (ID 12)
    ("linreg", "slope"): "Linear regression slope (positive=uptrend)",

    # HV (ID 13)
    ("hv", "hv"): "Historical volatility (annualized std dev of returns)",

    # Donchian (ID 14)
    ("donchian", "upper"): "Highest high over period (resistance)",
    ("donchian", "lower"): "Lowest low over period (support)",
    ("donchian", "basis"): "Midpoint of channel ((upper + lower) / 2)",

    # Floor Pivots (ID 15)
    ("floor_pivots", "pivot"): "Central pivot point",
    ("floor_pivots", "r1"): "Resistance level 1",
    ("floor_pivots", "s1"): "Support level 1",
    ("floor_pivots", "r2"): "Resistance level 2",
    ("floor_pivots", "s2"): "Support level 2",
    ("floor_pivots", "r3"): "Resistance level 3",
    ("floor_pivots", "s3"): "Support level 3",

    # Dynamic SR (ID 16)
    ("dynamic_sr", "resistance"): "Dynamic resistance level",
    ("dynamic_sr", "support"): "Dynamic support level",

    # Vol Targeting (ID 17)
    ("vol_targeting", "target_position"): "Target position size (fraction of capital)",

    # VRVP (ID 18)
    ("vrvp", "poc"): "Point of control (highest volume price level)",
    ("vrvp", "value_area_high"): "Value area high boundary",
    ("vrvp", "value_area_low"): "Value area low boundary",

    # RS Ratio (ID 19)
    ("rs_ratio", "rs_ratio"): "Relative strength ratio vs benchmark",

    # Correlation (ID 20)
    ("correlation", "correlation"): "Correlation coefficient (-1 to +1)",

    # Beta (ID 21)
    ("beta", "beta"): "Beta coefficient (sensitivity to benchmark)",

    # DD Price (ID 22)
    ("dd_price", "price_dd"): "Current price drawdown from peak (%)",
    ("dd_price", "price_dd_duration"): "Bars since price peak",

    # DD Per Trade (ID 23)
    ("dd_per_trade", "per_trade_dd"): "Max drawdown within current trade (%)",

    # DD Metrics (ID 24)
    ("dd_metrics", "max_dd"): "Maximum historical drawdown (%)",
    ("dd_metrics", "avg_dd"): "Average drawdown (%)",
    ("dd_metrics", "recovery_factor"): "Total return / max drawdown ratio",
}


def get_output_description(indicator_id: str, output_name: str) -> str:
    """Get human-readable description for an indicator output.
    Falls back to output_name if no description exists."""
    desc = OUTPUT_DESCRIPTIONS.get((indicator_id, output_name))
    if desc:
        return desc
    if indicator_id == "macd_tv":
        desc = OUTPUT_DESCRIPTIONS.get(("macd", output_name))
        if desc:
            return desc
    return output_name
