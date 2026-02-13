from fastmcp import FastMCP
import yfinance as yf
import pandas as pd
import sys
import os
from contextlib import contextmanager

# --- 0. SILENCE MANAGER ---
@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

mcp = FastMCP("Wall Street Publisher")

# --- 1. PROMPTS ---
@mcp.prompt()
def publisher_persona() -> str:
    return """You are a Portfolio Manager with an automated reporting system.
    1. When asked to scan stocks, ALWAYS use 'scan_and_publish'.
    2. Output the raw Markdown table in the chat so I can see it immediately.
    """

# --- 2. THE "BOSS" TOOL (Cloud Version) ---
@mcp.tool()
async def scan_and_publish(tickers: list[str]) -> str: # Changed: added async
    """
    Scans stocks, calculates DCF value, and ranks them.
    Args:
        tickers: List of symbols ['AAPL', 'NVDA']
    """
    results = []
    print(f"Scanning: {tickers}...") 
    
    for ticker in tickers:
        try:
            # yfinance is synchronous, but we run it inside the async tool.
            # in a heavy production app, you might offload this to a thread,
            # but for the Inspector, this works fine.
            with suppress_output():
                stock = yf.Ticker(ticker)
                info = stock.info
                
            fcf = info.get('freeCashflow')
            if not fcf:
                ocf = info.get('operatingCashflow', 0)
                capex = abs(info.get('capitalExpenditures', 0))
                fcf = ocf - capex
            
            if not fcf or fcf <= 0: continue

            shares = info.get('sharesOutstanding', 1)
            current_price = info.get('currentPrice', 0)
            
            # Assumptions (Boss Level: 40% Growth for Tech)
            growth_rate = 0.40 
            decay_rate = 0.05
            discount_rate = 0.10
            terminal_mult = 25
            
            # DCF Math
            current_fcf = fcf
            future_fcf = []
            g = growth_rate
            for _ in range(5):
                current_fcf = current_fcf * (1 + g)
                future_fcf.append(current_fcf)
                g = max(0.05, g - decay_rate)
                
            term_val = future_fcf[-1] * terminal_mult
            pv_fcf = sum([f / ((1 + discount_rate) ** (i+1)) for i, f in enumerate(future_fcf)])
            pv_term = term_val / ((1 + discount_rate) ** 5)
            
            fair_value = (pv_fcf + pv_term) / shares
            upside = ((fair_value - current_price) / current_price) * 100
            
            results.append({
                'Ticker': ticker,
                'Price': current_price,
                'Fair Value': fair_value,
                'Upside': upside
            })
            
        except Exception:
            continue

    # Sort results
    results.sort(key=lambda x: x['Upside'], reverse=True)
    
    # --- GENERATE MARKDOWN FOR CHAT ---
    report = f"✅ **CLOUD SCAN COMPLETE**\n\n"
    report += f"| {'Ticker':<6} | {'Price':<8} | {'Fair Value':<10} | {'Upside %':<8} |\n"
    report += f"|{'-'*8}|{'-'*10}|{'-'*12}|{'-'*10}|\n"
    
    for r in results:
        report += f"| {r['Ticker']:<6} | ${r['Price']:<7.2f} | ${r['Fair Value']:<9.2f} | {r['Upside']:>7.1f}% |\n"
        
    return report

if __name__ == "__main__":
    # Changed: Explicitly run with Streamable HTTP transport on port 8000
    # This makes the server accept Streamable-HTTP requests at the same `/sse` URL
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)