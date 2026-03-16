import streamlit as st
import yfinance as yf
import feedparser
import anthropic
import pandas as pd
from datetime import datetime, timedelta
import json
import re
import math

# ── Config ─────────────────────────────────────────────────────────────────────
TICKER      = "NBIS"
PEERS       = ["CRWV", "SMCI", "AMD", "NVDA"]
MACRO_SYMS  = {"^VIX": "VIX", "^TNX": "US 10Y", "^IXIC": "NASDAQ", "SMH": "Semis ETF"}

st.set_page_config(page_title="NBIS Research", page_icon="◆", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .block-container { padding: 0.8rem 1rem 3rem 1rem; max-width: 720px; }
    h1 { font-size: 1.5rem !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h2 { font-size: 1.15rem !important; font-weight: 600 !important; letter-spacing: -0.01em; }
    h3 { font-size: 1rem !important; font-weight: 600 !important; }
    .ticker-header { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.03em; }
    .price-big { font-size: 1.8rem; font-weight: 600; }
    .conviction-box {
        border-radius: 8px; padding: 16px; margin: 8px 0;
        border-left: 4px solid; text-align: center;
    }
    .bull { background: #0a2e1a; border-color: #22c55e; color: #22c55e; }
    .bear { background: #2e0a0a; border-color: #ef4444; color: #ef4444; }
    .neutral { background: #1a1a2e; border-color: #94a3b8; color: #94a3b8; }
    .score-bar { height: 6px; border-radius: 3px; margin: 2px 0 8px 0; }
    .morning-note {
        background: #111827; border: 1px solid #1f2937; border-radius: 8px;
        padding: 16px; margin: 8px 0; font-size: 0.92rem; line-height: 1.6;
    }
    .peer-table { font-size: 0.82rem; }
    .data-label { color: #6b7280; font-size: 0.75rem; text-transform: uppercase;
                  letter-spacing: 0.05em; font-weight: 500; }
    .data-value { font-size: 1rem; font-weight: 600; }
    .section-divider { border-top: 1px solid #1f2937; margin: 16px 0; }
    .risk-item { background: #1c1917; border-left: 3px solid #f59e0b;
                 padding: 8px 12px; margin: 4px 0; border-radius: 0 6px 6px 0; font-size: 0.85rem; }
    .catalyst-item { background: #0c1f0c; border-left: 3px solid #22c55e;
                     padding: 8px 12px; margin: 4px 0; border-radius: 0 6px 6px 0; font-size: 0.85rem; }
    .subsidiary-card { background: #111827; border: 1px solid #1f2937;
                       border-radius: 8px; padding: 12px; margin: 6px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_mcap(v):
    if not v: return "N/A"
    if v >= 1e12: return f"${v/1e12:.1f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"

def fmt_pct(v):
    if v is None: return "N/A"
    return f"{v:+.1f}%"

def conviction_class(score):
    if score >= 4:  return "bull"
    if score <= -4: return "bear"
    return "neutral"

def conviction_label(score):
    if score >= 7:  return "STRONG BUY"
    if score >= 4:  return "BUY"
    if score >= 1:  return "LEAN BULLISH"
    if score >= -1: return "NEUTRAL"
    if score >= -4: return "LEAN BEARISH"
    if score >= -7: return "SELL"
    return "STRONG SELL"

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def score_bar_html(score, max_score=2):
    pct = int((score + max_score) / (2 * max_score) * 100)
    pct = max(5, min(95, pct))
    if score >= 1:   color = "#22c55e"
    elif score >= 0: color = "#84cc16"
    elif score >= -1: color = "#f59e0b"
    else:            color = "#ef4444"
    return f'<div style="background:#1f2937;border-radius:3px;height:6px;"><div style="background:{color};width:{pct}%;height:6px;border-radius:3px;"></div></div>'


# ── Data Layer ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_all_data():
    # NBIS core data
    nbis = yf.Ticker(TICKER)
    info = nbis.info
    hist = nbis.history(period="6mo")

    price      = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or price
    change     = price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0

    ma20  = float(hist["Close"].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else None
    ma50  = float(hist["Close"].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else None
    ma200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else None
    rsi   = float(calc_rsi(hist["Close"]).iloc[-1]) if len(hist) >= 15 else None

    # Support / resistance from recent 3mo
    recent = hist["Close"].tail(60)
    support    = float(recent.min()) if len(recent) > 0 else None
    resistance = float(recent.max()) if len(recent) > 0 else None

    # Volume analysis
    vol       = info.get("volume", 0)
    avg_vol   = info.get("averageVolume", vol)
    vol_ratio = vol / avg_vol if avg_vol else 1.0

    # 52w position
    w52_high = info.get("fiftyTwoWeekHigh", 0)
    w52_low  = info.get("fiftyTwoWeekLow", 0)
    w52_pos  = ((price - w52_low) / (w52_high - w52_low) * 100) if (w52_high - w52_low) > 0 else 50

    stock = {
        "price": price, "prev_close": prev_close, "change": change,
        "change_pct": change_pct, "volume": vol, "avg_volume": avg_vol,
        "vol_ratio": vol_ratio, "week52_high": w52_high, "week52_low": w52_low,
        "week52_pos": w52_pos, "market_cap": info.get("marketCap"),
        "ma20": ma20, "ma50": ma50, "ma200": ma200, "rsi": rsi,
        "support": support, "resistance": resistance,
        "pe_ratio": info.get("trailingPE"), "fwd_pe": info.get("forwardPE"),
        "ev_revenue": info.get("enterpriseToRevenue"),
        "revenue_growth": info.get("revenueGrowth"),
        "sector": info.get("sector", "Technology"),
    }

    # Peer data
    peers = {}
    for sym in PEERS:
        try:
            t = yf.Ticker(sym)
            pi = t.info
            ph = t.history(period="5d")
            p_price = pi.get("currentPrice") or pi.get("regularMarketPrice") or 0
            p_prev  = pi.get("previousClose") or pi.get("regularMarketPreviousClose") or p_price
            p_chg   = ((p_price - p_prev) / p_prev * 100) if p_prev else 0
            peers[sym] = {
                "price": p_price, "change_pct": p_chg,
                "market_cap": pi.get("marketCap"),
                "ev_revenue": pi.get("enterpriseToRevenue"),
                "fwd_pe": pi.get("forwardPE"),
                "revenue_growth": pi.get("revenueGrowth"),
            }
        except Exception:
            peers[sym] = {"price": 0, "change_pct": 0, "market_cap": 0}

    # Macro data
    macro = {}
    for sym, label in MACRO_SYMS.items():
        try:
            t = yf.Ticker(sym)
            mh = t.history(period="5d")
            if len(mh) >= 2:
                curr = float(mh["Close"].iloc[-1])
                prev = float(mh["Close"].iloc[-2])
                macro[label] = {"value": curr, "change_pct": ((curr - prev) / prev * 100)}
            else:
                macro[label] = {"value": float(mh["Close"].iloc[-1]) if len(mh) > 0 else 0, "change_pct": 0}
        except Exception:
            macro[label] = {"value": 0, "change_pct": 0}

    # ── Price target model ──────────────────────────────────────────────────
    # Blended approach:
    # 1. Analyst consensus targets (anchored to real Wall Street estimates)
    # 2. Revenue multiple valuation (2026 guidance * peer EV/Rev, adjusted for shares)
    # 3. Historical volatility for confidence bands
    #
    # Analyst targets (from research):
    #   Low: $108 (GS, pre-NVIDIA deal), Avg: ~$150, High: $232 (Northland)
    #   Post-NVIDIA deal adjusted avg: ~$160
    # 2026 revenue guidance: $3.0-3.4B midpoint $3.2B
    # Non-core assets: ~$7.5B (ClickHouse $4.2B + Avride $2.2B + others)

    analyst_low    = 126   # Morgan Stanley (most recent conservative)
    analyst_avg    = 152   # consensus average
    analyst_high   = 232   # Northland Capital

    # Revenue multiple model: EV = Rev * multiple, then adjust for cash/debt/shares
    rev_2026_mid   = 3.2e9   # midpoint of $3.0-3.4B guidance
    shares_out     = 252e6   # ~218M float + warrants etc
    net_cash       = 3.5e9   # cash on hand
    non_core_value = 7.5e9   # ClickHouse + Avride + TripleTen + Toloka

    # Peer EV/Revenue multiples (infrastructure/cloud at hypergrowth):
    # Conservative: 8x, Base: 12x, Aggressive: 18x
    def rev_model(multiple):
        ev = rev_2026_mid * multiple
        equity = ev + net_cash + non_core_value
        return equity / shares_out

    rev_conservative = rev_model(8)    # ~$145
    rev_base         = rev_model(12)   # ~$196
    rev_aggressive   = rev_model(18)   # ~$272

    # Historical daily volatility → annualized
    daily_returns = hist["Close"].pct_change().dropna()
    daily_vol = float(daily_returns.std()) if len(daily_returns) > 20 else 0.04
    annual_vol = daily_vol * (252 ** 0.5)

    # Blended price targets per horizon
    # Weight: 50% analyst consensus path, 30% revenue model, 20% momentum/mean-reversion
    # Discount longer horizons toward base case, apply vol-scaled uncertainty
    price_targets = {}
    for months, key in [(1, "1m"), (3, "3m"), (6, "6m"), (12, "12m")]:
        t_frac = months / 12.0

        # Analyst path: interpolate from current price toward analyst_avg over 12m
        analyst_target = price + (analyst_avg - price) * t_frac

        # Revenue model path: interpolate toward rev_base over 12m
        # (as market increasingly prices in 2026 guidance execution)
        rev_target = price + (rev_base - price) * t_frac

        # Momentum component: extrapolate recent 20d trend
        if len(hist) >= 20:
            recent_20d_return = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-20] - 1)
            # Decay momentum over time (less reliable further out)
            momentum_target = price * (1 + recent_20d_return * min(months, 3) * 0.3)
        else:
            momentum_target = price

        # Blended target
        blended = (analyst_target * 0.50) + (rev_target * 0.30) + (momentum_target * 0.20)

        # Volatility-based confidence (1 sigma move for the period)
        vol_move = daily_vol * math.sqrt(months * 21) * price

        price_targets[key] = {
            "target": round(blended, 2),
            "pct": round((blended / price - 1) * 100, 1),
            "vol_1sigma": round(vol_move, 2),
            "low_1sigma": round(blended - vol_move, 2),
            "high_1sigma": round(blended + vol_move, 2),
            "method_note": f"50% analyst consensus (${analyst_avg}), 30% rev model (${rev_base:.0f} @ 12x 2026E rev), 20% momentum"
        }

    stock["price_targets"] = price_targets
    stock["annual_vol"] = round(annual_vol * 100, 1)
    stock["rev_model"] = {
        "conservative": round(rev_conservative, 0),
        "base": round(rev_base, 0),
        "aggressive": round(rev_aggressive, 0),
    }

    return stock, peers, macro


def fetch_news():
    feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NBIS&region=US&lang=en-US",
        "https://news.google.com/rss/search?q=Nebius+Group+AI&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=%22Nebius%22+OR+%22NBIS%22+stock&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=AI+datacenter+GPU+infrastructure+2025&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Toloka+OR+TripleTen+OR+Avride+AI&hl=en-US&gl=US&ceid=US:en",
    ]
    articles = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:5]:
                articles.append({
                    "title":     e.get("title", ""),
                    "summary":   e.get("summary", "")[:400],
                    "published": e.get("published", ""),
                    "link":      e.get("link", ""),
                })
        except Exception:
            pass
    return articles[:25]


def run_analysis(stock, peers, macro, news):
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("Mangler ANTHROPIC_API_KEY.")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    # Build context
    tech_signal = "neutral"
    if stock["ma20"] and stock["price"]:
        if stock["price"] > stock["ma20"] and (stock["ma50"] is None or stock["price"] > stock["ma50"]):
            tech_signal = "bullish (above MA20 & MA50)"
        elif stock["price"] > stock["ma20"]:
            tech_signal = "mildly bullish (above MA20, below MA50)"
        else:
            tech_signal = "bearish (below MA20)"

    news_text = "\n".join([f"- {a['title']}" for a in news[:20]])

    peer_text = ""
    for sym, p in peers.items():
        peer_text += f"  {sym}: ${p['price']:.2f} ({fmt_pct(p.get('change_pct'))} today), MCap {fmt_mcap(p.get('market_cap'))}, EV/Rev {p.get('ev_revenue', 'N/A')}x, Fwd P/E {p.get('fwd_pe', 'N/A')}\n"

    macro_text = ""
    for label, m in macro.items():
        macro_text += f"  {label}: {m['value']:.2f} ({fmt_pct(m['change_pct'])})\n"

    # Pre-format values to avoid nested f-string issues
    ma20_str  = f"${stock['ma20']:.2f}" if stock['ma20'] else "N/A"
    ma50_str  = f"${stock['ma50']:.2f}" if stock['ma50'] else "N/A"
    ma200_str = f"${stock['ma200']:.2f}" if stock['ma200'] else "N/A"
    rsi_str   = f"{stock['rsi']:.1f}" if stock['rsi'] else "N/A"
    sup_str   = f"${stock['support']:.2f}" if stock['support'] else "N/A"
    res_str   = f"${stock['resistance']:.2f}" if stock['resistance'] else "N/A"
    rev_growth_str = fmt_pct(stock.get('revenue_growth', 0) * 100 if stock.get('revenue_growth') else None)

    prompt = f"""You are the lead equity research analyst at Goldman Sachs covering AI infrastructure.
Write a daily institutional research brief for NBIS (Nebius Group N.V.).

═══════════════════════════════════════════════════════════════════════
PERMANENT KNOWLEDGE BASE — NBIS / NEBIUS GROUP
(Use this as the foundation for ALL analysis. Today's news/data adds signal on top.)
═══════════════════════════════════════════════════════════════════════

COMPANY OVERVIEW:
Nebius Group N.V. (NASDAQ: NBIS) is a full-stack AI cloud infrastructure company spun out
of Yandex. Core business is GPU-as-a-service for AI training and inference workloads.
HQ in Amsterdam. CEO: Arkady Volozh (founder of Yandex).

NVIDIA STRATEGIC PARTNERSHIP (announced March 11-12, 2026):
- NVIDIA invested $2B via pre-funded warrants (21.07M Class A shares at $0.0001)
- This is NOT just an investment — it's a strategic alliance:
  * EARLY ACCESS to next-gen Rubin platform, Vera CPUs, Blackwell Ultra, BlueField storage
  * Rubin offers ~10x cost-effectiveness over Blackwell for inference
  * Joint target: 5 GW of NVIDIA-powered capacity by 2030
  * Co-designing high-density liquid-cooled AI Factory data centers
- Implication: In a world of chip scarcity, Nebius is at the FRONT of the queue for
  NVIDIA's best silicon. This is a massive competitive moat.

FINANCIALS (as of Q4 2025 / FY2025):
- Q4 2025 revenue: $227.7M (+547% YoY, +56% QoQ)
- FY 2025 revenue: $529.8M (+351% YoY)
- Q4 core AI cloud growth: +830% YoY
- Q4 adjusted EBITDA margin: 24% (up from 19% Q3)
- Q4 operating cash flow: $834M
- Cash on hand: $3.0-3.7B
- ARR exiting 2025: ~$1.25B
- 2026 revenue guidance: $3.0-3.4B (470-540% YoY growth)
- 2026 ARR target: $7-9B
- 2026 adj. EBITDA margin target: ~40%
- 2026 CapEx guidance: $16-20B
- Active pipeline: >$4B, deal sizes increasing, contract terms lengthening
- Net loss Q4 2025: -$249.6M (investment phase, expected)

MAJOR CONTRACTS:
- Microsoft: $19.4B, 5-year deal for 100K+ GB300 chips
- Meta: $3B, 5-year infrastructure deal
- ALL current capacity is SOLD OUT

INFRASTRUCTURE:
- Current: ~170 MW active (end of 2025)
- 2026 target: 800 MW - 1 GW connected capacity
- Contracted power target: 2.5 GW by end 2026 (raised from 1 GW)
- Data centers: 7 sites (2025) → 16 sites (2026)
  * Finland (ex-Yandex), Iceland, Paris (Equinix), New Jersey (300 MW w/ DataOne)
  * Independence, Missouri: 1.2 GW campus, 400 acres (approved March 2026)
- Efficiency: Claims 3x more compute per megawatt vs competitors
- GPU offerings: GB300 NVL72, GB200 NVL72, B300, B200, H200, H100 + InfiniBand

KEY PRODUCTS:
- Token Factory: Managed inference endpoints, priced by token throughput/latency
- Aether (AI Cloud 3.0): SOC 2 Type II, HIPAA, ISO 27001 certified
- Competitive H100 pricing at ~$2.10/hr vs CoreWeave ~$4.76/hr

SUBSIDIARIES & EQUITY STAKES (combined non-core value ~$7.5-8B):
1. Nebius AI Cloud — Core business (~90% of revenue), full-stack GPU cloud
2. Avride — Autonomous vehicles & delivery robots. Implied value ~$2.2-2.3B. IPO candidate.
3. TripleTen — EdTech bootcamp (US & Latin America). Named Best Software Bootcamp by Fortune.
   Revenue 2x+ in 2025 (+251% YoY in FY2024). Growing fast.
4. Toloka — AI data labeling. Deconsolidated Q2 2025. Bezos Expeditions invested.
   Nebius retains majority economic interest. FY2024 revenue $26.4M (+138% YoY).
5. ClickHouse — ~28% equity stake. Real-time analytics DB. Series D at ~$15B valuation
   (Jan 2026, raised $400M). Customers: Anthropic, Meta. Implied stake value ~$4.2B. IPO candidate.

AI INFERENCE MARKET (critical context):
- Inference is rising to ~2/3 of all AI compute by 2026 (from 1/3 in 2023)
- Inference chip market: ~$20B (2025) → $50B+ (2026) — Deloitte
- Total AI inference market: $106B (2025) → $255B by 2030 (19.2% CAGR)
- AWS says up to 90% of AI workloads/spend will be inference
- Jensen Huang: inference demand "100x more" than initially expected
- 2026 is the "breakout year" for AI inferencing
- Nebius Token Factory is purpose-built for this wave

ANALYST CONSENSUS:
- Strong Buy (6 Buy, 1 Hold, 0 Sell)
- Average price target: ~$150-152
- High target: $232 (Northland Capital, Nov 2025)
- Low target: $108 (Goldman Sachs, Sept 2025 — likely outdated pre-NVIDIA deal)
- Recent: Compass Point $150, Morgan Stanley $126, JMP Securities $175

SHARE STRUCTURE:
- Float: ~218M shares (86.6% of total)
- Short interest: ~19% of float (elevated — short squeeze potential)
- Institutional ownership: ~22-40% and growing
- Largest holder: BlackRock (3.73%)
- NVIDIA warrants: 21.07M additional shares (not yet exercised)

KEY RISKS:
1. Massive CapEx: $16-20B in 2026, ~60% funded — execution & financing risk
2. H2-weighted revenue: Most 2026 revenue depends on data centers coming online on schedule
3. Continued EBIT losses through 2026 due to growth investment
4. GPU price commoditization: H100 rental rates down 60-75% from peak
5. Customer concentration: Microsoft and Meta are bulk of backlog
6. Short interest at 19% creates volatility
7. Geopolitical: Yandex heritage may create perception issues with some customers

UPCOMING CATALYSTS:
- April 29, 2026: Q1 2026 earnings
- H2 2026: Missouri data center power delivery
- 2026: NVIDIA Rubin/Vera chip deliveries begin
- TBD: ClickHouse IPO (widely expected)
- TBD: Avride potential IPO/spin-off

COMPETITIVE LANDSCAPE:
vs CoreWeave (CRWV): CRWV larger backlog ($56B) but 4.8x debt-to-equity with $34B off-balance-sheet leases.
  Nebius has far stronger balance sheet ($6B+ cash vs CRWV's leverage). Nebius prices 50%+ lower.
vs Lambda Labs: Smaller, developer-focused, less enterprise-grade
vs Hyperscalers (AWS/Azure/GCP): Nebius competes on price, specialized AI optimization, and latency

═══════════════════════════════════════════════════════════════════════
TODAY'S LIVE DATA
═══════════════════════════════════════════════════════════════════════

NBIS: ${stock['price']:.2f} ({fmt_pct(stock['change_pct'])} today)
52-Week: ${stock['week52_low']:.2f} – ${stock['week52_high']:.2f} ({stock['week52_pos']:.0f}th percentile)
Market Cap: {fmt_mcap(stock['market_cap'])}
Volume: {stock['volume']:,} ({stock['vol_ratio']:.1f}x avg)
EV/Revenue: {stock.get('ev_revenue', 'N/A')}x | Fwd P/E: {stock.get('fwd_pe', 'N/A')} | Rev Growth: {rev_growth_str}

TECHNICALS:
Signal: {tech_signal}
RSI(14): {rsi_str}
MA20: {ma20_str} | MA50: {ma50_str} | MA200: {ma200_str}
Support: {sup_str} | Resistance: {res_str}

PEERS TODAY:
{peer_text}

MACRO TODAY:
{macro_text}

NEWS FLOW (last 48h):
{news_text}

═══════════════════════════════════════════════════════════════════════
ANALYSIS INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

You MUST incorporate the permanent knowledge base above into your analysis.
DO NOT treat this as a generic stock — you have deep fundamental knowledge of NBIS.

For FAIR VALUE, use these approaches and SHOW YOUR REASONING in the thesis fields:
1. Revenue multiple: Apply peer EV/Revenue multiples to 2026 guidance ($3.0-3.4B)
2. ARR-based: Value the $7-9B ARR target at appropriate SaaS/infra multiples
3. Sum-of-parts: Core AI cloud + ClickHouse stake ($4.2B) + Avride ($2.2B) + TripleTen + Toloka
4. Consider that $108 fair value would be BELOW the old GS target which was set BEFORE the NVIDIA deal
5. Factor in the inference market TAM explosion and Nebius's positioning

For SCORING, weight fundamentals and forward outlook MORE heavily than technicals.
Technicals are useful but secondary for a hypergrowth infrastructure company.

For SUBSIDIARIES, assess each one's trajectory and value contribution.

Be intellectually honest about risks. But also be honest about the asymmetric upside.

Return ONLY valid JSON:
{{
  "morning_note": "<4-6 sentence institutional executive summary. Be specific — cite revenue numbers, the NVIDIA deal, inference market dynamics, peer comparisons. This should read like a GS morning note.>",
  "conviction": "<STRONG BUY | BUY | LEAN BULLISH | NEUTRAL | LEAN BEARISH | SELL | STRONG SELL>",
  "scores": {{
    "macro_tailwind":        {{"score": <-2 to 2>, "detail": "<cite VIX, yields, risk sentiment, AI capex cycle>"}},
    "sector_momentum":       {{"score": <-2 to 2>, "detail": "<inference TAM, datacenter buildout, peer performance>"}},
    "company_fundamentals":  {{"score": <-2 to 2>, "detail": "<revenue trajectory, NVIDIA deal, backlog, margins>"}},
    "technical_setup":       {{"score": <-2 to 2>, "detail": "<RSI, MA structure, support/resistance>"}},
    "news_catalyst":         {{"score": <-2 to 2>, "detail": "<today's specific news impact>"}},
    "valuation_vs_peers":    {{"score": <-2 to 2>, "detail": "<EV/Rev vs CRWV/SMCI, growth-adjusted, SOTP value>"}},
    "institutional_flow":    {{"score": <-2 to 2>, "detail": "<volume, short interest, NVIDIA investment signal>"}}
  }},
  "total_score": <sum of all 7 scores, -14 to 14>,
  "fair_value": {{
    "bear": {{"price": <number>, "probability": <0-100>, "thesis": "<specific bear case with numbers>"}},
    "base": {{"price": <number>, "probability": <0-100>, "thesis": "<specific base case with valuation method>"}},
    "bull": {{"price": <number>, "probability": <0-100>, "thesis": "<specific bull case with upside drivers>"}}
  }},
  "probability_weighted_value": <weighted average>,
  "subsidiaries": {{
    "nebius_cloud":  {{"status": "<current state with specifics>", "signal": "<bullish|neutral|bearish>", "implied_value": "<estimate>"}},
    "toloka":        {{"status": "<state + Bezos investment context>", "signal": "<bullish|neutral|bearish>", "implied_value": "<estimate>"}},
    "tripletens":    {{"status": "<growth trajectory>", "signal": "<bullish|neutral|bearish>", "implied_value": "<estimate>"}},
    "avride":        {{"status": "<autonomous driving progress>", "signal": "<bullish|neutral|bearish>", "implied_value": "<estimate>"}},
    "clickhouse":    {{"status": "<$15B valuation, 28% stake>", "signal": "<bullish|neutral|bearish>", "implied_value": "<estimate>"}}
  }},
  "risks": [
    {{"risk": "<specific risk>", "severity": "<HIGH|MEDIUM|LOW>", "probability": "<HIGH|MEDIUM|LOW>", "mitigation": "<what could offset this>"}},
    {{"risk": "<specific risk>", "severity": "<HIGH|MEDIUM|LOW>", "probability": "<HIGH|MEDIUM|LOW>", "mitigation": "<what could offset this>"}},
    {{"risk": "<specific risk>", "severity": "<HIGH|MEDIUM|LOW>", "probability": "<HIGH|MEDIUM|LOW>", "mitigation": "<what could offset this>"}},
    {{"risk": "<specific risk>", "severity": "<HIGH|MEDIUM|LOW>", "probability": "<HIGH|MEDIUM|LOW>", "mitigation": "<what could offset this>"}},
    {{"risk": "<specific risk>", "severity": "<HIGH|MEDIUM|LOW>", "probability": "<HIGH|MEDIUM|LOW>", "mitigation": "<what could offset this>"}}
  ],
  "catalysts": [
    {{"catalyst": "<specific catalyst>", "timeline": "<date/quarter>", "impact": "<HIGH|MEDIUM|LOW>", "detail": "<why this matters>"}},
    {{"catalyst": "<specific catalyst>", "timeline": "<date/quarter>", "impact": "<HIGH|MEDIUM|LOW>", "detail": "<why this matters>"}},
    {{"catalyst": "<specific catalyst>", "timeline": "<date/quarter>", "impact": "<HIGH|MEDIUM|LOW>", "detail": "<why this matters>"}},
    {{"catalyst": "<specific catalyst>", "timeline": "<date/quarter>", "impact": "<HIGH|MEDIUM|LOW>", "detail": "<why this matters>"}},
    {{"catalyst": "<specific catalyst>", "timeline": "<date/quarter>", "impact": "<HIGH|MEDIUM|LOW>", "detail": "<why this matters>"}}
  ],
  "key_levels": {{
    "immediate_support": <price>,
    "strong_support": <price>,
    "immediate_resistance": <price>,
    "breakout_target": <price>
  }},
  "what_to_watch": "<3-4 specific things to monitor this week, with reasoning>"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text

    # Extract JSON robustly
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        st.error("Kunne ikke finde JSON i Claude's svar.")
        return None

    json_str = match.group()
    # Clean common JSON issues
    json_str = re.sub(r',\s*}', '}', json_str)   # trailing commas before }
    json_str = re.sub(r',\s*]', ']', json_str)   # trailing commas before ]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.error(f"JSON parse fejl: {e}")
        st.code(json_str[:500], language="json")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(f'<div class="ticker-header">◆ NBIS</div>', unsafe_allow_html=True)
st.caption(f"Nebius Group N.V. · AI Infrastructure · {st.session_state.get('last_update', 'Ikke opdateret endnu')}")

if st.button("Opdater analyse", use_container_width=True, type="primary"):
    with st.spinner("Henter markedsdata, nyheder og kører analyse..."):
        try:
            stock, peers, macro = fetch_all_data()
            news = fetch_news()
            result = run_analysis(stock, peers, macro, news)
            if result:
                st.session_state["stock"]       = stock
                st.session_state["peers"]       = peers
                st.session_state["macro"]       = macro
                st.session_state["news"]        = news
                st.session_state["result"]      = result
                st.session_state["last_update"] = datetime.now().strftime("%d. %b %Y — %H:%M")
                st.cache_data.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Fejl: {e}")

if "result" not in st.session_state:
    st.markdown("---")
    st.markdown("Tryk **Opdater analyse** for at starte.")
    st.stop()

stock  = st.session_state["stock"]
peers  = st.session_state["peers"]
macro  = st.session_state["macro"]
news   = st.session_state["news"]
result = st.session_state["result"]
ts     = result.get("total_score", 0)

# ── Price bar ──────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2 = st.columns([2, 1])
with c1:
    arrow = "▲" if stock["change"] >= 0 else "▼"
    color = "#22c55e" if stock["change"] >= 0 else "#ef4444"
    st.markdown(f'<span class="price-big">${stock["price"]:.2f}</span> <span style="color:{color};font-size:1.1rem;">{arrow} {stock["change_pct"]:+.2f}%</span>', unsafe_allow_html=True)
    st.caption(f'Vol: {stock["volume"]:,} ({stock["vol_ratio"]:.1f}x avg) · MCap: {fmt_mcap(stock["market_cap"])}')
with c2:
    conv = result.get("conviction", "NEUTRAL")
    css_class = conviction_class(ts)
    st.markdown(f'<div class="conviction-box {css_class}"><b>{conv}</b><br><span style="font-size:0.8rem;">Score: {ts:+d}/14</span></div>', unsafe_allow_html=True)

# ── Morning Note ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Morning Note")
st.markdown(f'<div class="morning-note">{result.get("morning_note", "")}</div>', unsafe_allow_html=True)

# ── Score Breakdown ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Score Breakdown")

score_labels = {
    "macro_tailwind":       "Makro",
    "sector_momentum":      "AI/DC Sektor",
    "company_fundamentals": "Fundamentals",
    "technical_setup":      "Teknisk",
    "news_catalyst":        "Nyheder",
    "valuation_vs_peers":   "Værdiansættelse",
    "institutional_flow":   "Institutional Flow",
}

for key, label in score_labels.items():
    s_data = result.get("scores", {}).get(key, {})
    s = s_data.get("score", 0) if isinstance(s_data, dict) else 0
    detail = s_data.get("detail", "") if isinstance(s_data, dict) else ""
    st.markdown(f"**{label}** `{s:+d}`")
    st.markdown(score_bar_html(s), unsafe_allow_html=True)
    if detail:
        st.caption(detail)

# ── Fair Value ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Fair Value Estimat")

fv   = result.get("fair_value", {})
pwv  = result.get("probability_weighted_value", 0)
curr = stock["price"]
upside = ((pwv - curr) / curr * 100) if curr else 0

st.markdown(f"**Probability-Weighted Fair Value: ${pwv:.0f}** ({upside:+.0f}% vs current)")

for case, emoji in [("bear", "🐻"), ("base", "📊"), ("bull", "🚀")]:
    cv = fv.get(case, {})
    if isinstance(cv, dict):
        p = cv.get("price", 0)
        prob = cv.get("probability", 0)
        thesis = cv.get("thesis", "")
        diff = ((p - curr) / curr * 100) if curr else 0
        st.markdown(f"{emoji} **{case.title()}** — ${p:.0f} ({diff:+.0f}%) · {prob}% sandsynlighed")
        st.caption(thesis)

# ── Price Targets ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Kursmål")

pt = stock.get("price_targets", {})
rm = stock.get("rev_model", {})

# Show the 4 horizons side by side
st.markdown(f"**Metode:** 50% analyst consensus ($152), 30% revenue model (${rm.get('base', 0):.0f} @ 12x 2026E rev + SOTP), 20% momentum")
st.caption(f"Annualiseret volatilitet: {stock.get('annual_vol', 0)}% · Revenue model range: ${rm.get('conservative', 0):.0f} (8x) – ${rm.get('base', 0):.0f} (12x) – ${rm.get('aggressive', 0):.0f} (18x)")

st.markdown("")
c1, c2, c3, c4 = st.columns(4)
for col, (key, label) in zip([c1, c2, c3, c4], [("1m", "1 måned"), ("3m", "3 måneder"), ("6m", "6 måneder"), ("12m", "12 måneder")]):
    t = pt.get(key, {})
    target = t.get("target", 0)
    pct = t.get("pct", 0)
    low = t.get("low_1sigma", 0)
    high = t.get("high_1sigma", 0)
    color = "#22c55e" if pct >= 0 else "#ef4444"
    with col:
        st.markdown(f'<span class="data-label">{label}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="data-value" style="color:{color};">${target:.0f}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:{color};font-size:0.9rem;">{pct:+.1f}%</span>', unsafe_allow_html=True)
        st.caption(f"${low:.0f} – ${high:.0f}")
        st.caption("1σ range")

# ── Technicals ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Teknisk Oversigt")

kl = result.get("key_levels", {})
c1, c2 = st.columns(2)
with c1:
    if stock.get("rsi"):
        st.markdown(f'<span class="data-label">RSI (14)</span><br><span class="data-value">{stock["rsi"]:.1f}</span>', unsafe_allow_html=True)
    if stock.get("ma20"):
        st.markdown(f'<span class="data-label">MA20</span><br><span class="data-value">${stock["ma20"]:.2f}</span>', unsafe_allow_html=True)
    if stock.get("ma50"):
        st.markdown(f'<span class="data-label">MA50</span><br><span class="data-value">${stock["ma50"]:.2f}</span>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<span class="data-label">Support</span><br><span class="data-value">${kl.get("immediate_support", 0):.2f}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="data-label">Resistance</span><br><span class="data-value">${kl.get("immediate_resistance", 0):.2f}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="data-label">Breakout Target</span><br><span class="data-value">${kl.get("breakout_target", 0):.2f}</span>', unsafe_allow_html=True)

st.caption(f"52-ugers range: ${stock['week52_low']:.2f} – ${stock['week52_high']:.2f} ({stock['week52_pos']:.0f}th percentile)")

# ── Peers ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Peer Sammenligning")

peer_rows = []
# Add NBIS first
peer_rows.append({
    "Ticker": "**NBIS**",
    "Pris": f"${stock['price']:.2f}",
    "Dag": fmt_pct(stock["change_pct"]),
    "MCap": fmt_mcap(stock["market_cap"]),
    "EV/Rev": f"{stock.get('ev_revenue', 'N/A')}x" if stock.get('ev_revenue') else "N/A",
    "Fwd P/E": f"{stock.get('fwd_pe', 'N/A')}",
})
for sym, p in peers.items():
    peer_rows.append({
        "Ticker": sym,
        "Pris": f"${p['price']:.2f}" if p.get('price') else "N/A",
        "Dag": fmt_pct(p.get("change_pct")),
        "MCap": fmt_mcap(p.get("market_cap")),
        "EV/Rev": f"{p.get('ev_revenue', 'N/A')}x" if p.get('ev_revenue') else "N/A",
        "Fwd P/E": f"{p.get('fwd_pe', 'N/A')}",
    })
st.dataframe(pd.DataFrame(peer_rows), hide_index=True, use_container_width=True)

# ── Macro ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Makro Miljø")

macro_cols = st.columns(len(macro))
for col, (label, m) in zip(macro_cols, macro.items()):
    color = "#22c55e" if m["change_pct"] >= 0 else "#ef4444"
    with col:
        st.markdown(f'<span class="data-label">{label}</span><br><span class="data-value">{m["value"]:.2f}</span><br><span style="color:{color};font-size:0.8rem;">{fmt_pct(m["change_pct"])}</span>', unsafe_allow_html=True)

# ── Subsidiaries ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Datterselskaber")

subs = result.get("subsidiaries", {})
sub_labels = {
    "nebius_cloud": ("Nebius AI Cloud", "☁️"),
    "clickhouse":   ("ClickHouse (28% stake)", "🗄️"),
    "avride":       ("Avride", "🚗"),
    "tripletens":   ("TripleTen", "🎓"),
    "toloka":       ("Toloka", "🏷️"),
}

for key, (name, icon) in sub_labels.items():
    s = subs.get(key, {})
    if isinstance(s, dict):
        signal = s.get("signal", "neutral")
        sig_color = {"bullish": "#22c55e", "neutral": "#94a3b8", "bearish": "#ef4444"}.get(signal, "#94a3b8")
        status = s.get("status", "")
        imp_val = s.get("implied_value", "")
        val_str = f" · {imp_val}" if imp_val else ""
        st.markdown(f'<div class="subsidiary-card">{icon} <b>{name}</b> <span style="color:{sig_color};float:right;">● {signal.upper()}</span><br><span style="color:#9ca3af;font-size:0.85rem;">{status}{val_str}</span></div>', unsafe_allow_html=True)

# ── Risks & Catalysts ─────────────────────────────────────────────────────────
st.markdown("---")
col_r, col_c = st.columns(2)

with col_r:
    st.markdown("## Risici")
    for r in result.get("risks", []):
        if isinstance(r, dict):
            sev = r.get("severity", "MEDIUM")
            sev_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "🟡")
            mitigation = r.get("mitigation", "")
            mit_str = f"<br><span style='color:#4b5563;font-size:0.72rem;'>Mitigation: {mitigation}</span>" if mitigation else ""
            st.markdown(f'<div class="risk-item">{sev_icon} {r.get("risk", "")}<br><span style="color:#6b7280;font-size:0.75rem;">Severity: {sev} · Probability: {r.get("probability", "")}</span>{mit_str}</div>', unsafe_allow_html=True)

with col_c:
    st.markdown("## Katalysatorer")
    for c in result.get("catalysts", []):
        if isinstance(c, dict):
            impact = c.get("impact", "MEDIUM")
            imp_icon = {"HIGH": "⚡", "MEDIUM": "📌", "LOW": "📎"}.get(impact, "📌")
            detail = c.get("detail", "")
            det_str = f"<br><span style='color:#4b5563;font-size:0.72rem;'>{detail}</span>" if detail else ""
            st.markdown(f'<div class="catalyst-item">{imp_icon} {c.get("catalyst", "")}<br><span style="color:#6b7280;font-size:0.75rem;">{c.get("timeline", "")} · Impact: {impact}</span>{det_str}</div>', unsafe_allow_html=True)

# ── What to Watch ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Denne Uge")
st.info(result.get("what_to_watch", ""))

# ── News ───────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📰 News Feed"):
    for a in news[:12]:
        st.markdown(f"**{a['title']}**")
        if a.get("published"):
            st.caption(a["published"])
        st.markdown("---")

# Footer
st.markdown("---")
st.caption("◆ Research genereret af Claude AI · Ikke finansiel rådgivning · Data fra Yahoo Finance")
