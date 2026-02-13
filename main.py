from fastmcp import FastMCP
import yfinance as yf
import pandas as pd
import sys
import os
from contextlib import contextmanager
from fpdf import FPDF
from datetime import datetime
import base64

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


def generate_pdf_report(results: list[dict], filename: str | None = None, return_bytes: bool = False):
    """Create a PDF report. Either save to `filename` and return path, or return bytes when `return_bytes=True`.

    - If `return_bytes` is True, the PDF bytes are returned (useful for in-memory download/base64).
    - If `return_bytes` is False, the file is saved under `reports/` and the path is returned.
    """
    # build PDF
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "Wall Street Publisher - Scan Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} (UTC)", ln=True)
    pdf.ln(4)

    # Summary table header
    pdf.set_font("Helvetica", "B", 11)
    col_widths = [30, 34, 40, 30]
    headers = ["Ticker", "Price", "Fair Value", "Upside %"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C")
    pdf.ln()

    # Summary table rows
    pdf.set_font("Helvetica", "", 10)
    for r in results:
        pdf.cell(col_widths[0], 7, str(r.get("Ticker", "")), border=1)
        pdf.cell(col_widths[1], 7, f"${r.get('Price',0):,.2f}", border=1, align="R")
        pdf.cell(col_widths[2], 7, f"${r.get('Fair Value',0):,.2f}", border=1, align="R")
        pdf.cell(col_widths[3], 7, f"{r.get('Upside',0):.1f}%", border=1, align="R")
        pdf.ln()

    pdf.ln(6)

    # Per-ticker detailed insights
    for r in results:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, f"{r['Ticker']} - Detailed DCF & Assumptions", ln=True)
        pdf.set_font("Helvetica", "", 10)
        assumptions = (
            f"Growth rate start: {r.get('growth_rate', 0):.0%} | "
            f"Decay rate: {r.get('decay_rate', 0):.0%} | "
            f"Discount rate: {r.get('discount_rate', 0):.0%} | "
            f"Terminal multiple: {r.get('terminal_mult', 0)}"
        )
        pdf.multi_cell(0, 5, assumptions)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(30, 6, "Year", border=1, align="C")
        pdf.cell(50, 6, "Projected FCF", border=1, align="C")
        pdf.cell(50, 6, "Discounted PV", border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)

        future = r.get("future_fcf", [])
        discount = r.get("discount_rate", 0.10)
        for i, fcf in enumerate(future, start=1):
            pv = fcf / ((1 + discount) ** i)
            pdf.cell(30, 6, f"Yr {i}", border=1)
            pdf.cell(50, 6, f"${fcf:,.2f}", border=1, align="R")
            pdf.cell(50, 6, f"${pv:,.2f}", border=1, align="R")
            pdf.ln()

        pdf.ln(2)
        pdf.cell(0, 5, f"Terminal value (post-year-5): ${r.get('term_val',0):,.2f}")
        pdf.ln(4)
        pdf.cell(0, 5, f"PV of projections: ${r.get('pv_fcf',0):,.2f} | PV of terminal: ${r.get('pv_term',0):,.2f}")
        pdf.ln(6)

    if return_bytes:
        # Return PDF bytes. fpdf.output(dest='S') may return str or bytearray depending on version.
        pdf_s = pdf.output(dest='S')
        if isinstance(pdf_s, (bytes, bytearray)):
            return bytes(pdf_s)
        return str(pdf_s).encode('latin-1')

    # Save file path
    os.makedirs("reports", exist_ok=True)
    if filename is None:
        filename = f"reports/scan_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.pdf"
    pdf.output(filename)
    return filename


mcp = FastMCP("Wall Street Publisher")

# --- 1. PROMPTS ---
@mcp.prompt()
def publisher_persona() -> str:
    return """You are a Portfolio Manager with an automated reporting system.
    1. When asked to scan stocks, ALWAYS use 'scan_and_publish'.
    2. Output the raw Markdown table in the chat so I can see it immediately.
    """


# Helper: core scan logic extracted so both tools can reuse it
def _scan_tickers(tickers: list[str]) -> list[dict]:
    results: list[dict] = []
    for ticker in tickers:
        try:
            with suppress_output():
                stock = yf.Ticker(ticker)
                info = stock.info

            fcf = info.get('freeCashflow')
            if not fcf:
                ocf = info.get('operatingCashflow', 0)
                capex = abs(info.get('capitalExpenditures', 0))
                fcf = ocf - capex

            if not fcf or fcf <= 0:
                continue

            shares = info.get('sharesOutstanding', 1)
            current_price = info.get('currentPrice', 0)

            # Assumptions (Boss Level: 40% Growth for Tech)
            growth_rate = 0.40
            decay_rate = 0.05
            discount_rate = 0.10
            terminal_mult = 25

            # DCF Math (5-year projection)
            current_fcf = fcf
            future_fcf: list[float] = []
            g = growth_rate
            for _ in range(5):
                current_fcf = current_fcf * (1 + g)
                future_fcf.append(current_fcf)
                g = max(0.05, g - decay_rate)

            term_val = future_fcf[-1] * terminal_mult
            pv_fcf = sum([f / ((1 + discount_rate) ** (i + 1)) for i, f in enumerate(future_fcf)])
            pv_term = term_val / ((1 + discount_rate) ** 5)

            fair_value = (pv_fcf + pv_term) / shares
            upside = ((fair_value - current_price) / current_price) * 100 if current_price else 0.0

            results.append({
                'Ticker': ticker,
                'Price': current_price,
                'Fair Value': fair_value,
                'Upside': upside,
                # detailed fields for PDF / insights
                'shares': shares,
                'fcf': fcf,
                'future_fcf': future_fcf,
                'pv_fcf': pv_fcf,
                'pv_term': pv_term,
                'term_val': term_val,
                'growth_rate': growth_rate,
                'decay_rate': decay_rate,
                'discount_rate': discount_rate,
                'terminal_mult': terminal_mult,
            })

        except Exception:
            continue

    results.sort(key=lambda x: x['Upside'], reverse=True)
    return results


# --- 2. THE "BOSS" TOOL (Cloud Version) ---
@mcp.tool()
async def scan_and_publish(tickers: list[str]) -> str: # Changed: added async
    """
    Scans stocks, calculates DCF value, ranks them, returns markdown and saves a PDF with detailed insights.
    Args:
        tickers: List of symbols ['AAPL', 'NVDA']
    """
    print(f"Scanning: {tickers}...")
    results = _scan_tickers(tickers)

    # --- GENERATE MARKDOWN FOR CHAT (quick view) ---
    report = f"✅ **CLOUD SCAN COMPLETE**\n\n"
    report += f"| {'Ticker':<6} | {'Price':<8} | {'Fair Value':<10} | {'Upside %':<8} |\n"
    report += f"|{'-'*8}|{'-'*10}|{'-'*12}|{'-'*10}|\n"

    for r in results:
        report += f"| {r['Ticker']:<6} | ${r['Price']:<7.2f} | ${r['Fair Value']:<9.2f} | {r['Upside']:>7.1f}% |\n"

    # --- DETAILED INSIGHTS (markdown) ---
    report += "\n---\n\n"
    report += "### Detailed insights\n\n"
    for r in results:
        report += f"**{r['Ticker']}** - Current: ${r['Price']:.2f} | Fair value: ${r['Fair Value']:.2f} | Upside: {r['Upside']:.1f}%\n\n"
        report += "**Assumptions:**\n"
        report += f"- Start growth: {r['growth_rate']:.0%}, decay per year: {r['decay_rate']:.0%}\n"
        report += f"- Discount rate: {r['discount_rate']:.0%}, terminal multiple: {r['terminal_mult']}\n\n"
        report += "**5‑year projection (FCF)**:\n"
        report += "| Year | Projected FCF |\n"
        report += "|---:|---:|\n"
        for i, fcf in enumerate(r['future_fcf'], start=1):
            report += f"| {i} | ${fcf:,.2f} |\n"
        report += "\n"

    # --- SAVE PDF REPORT ---
    pdf_path = generate_pdf_report(results)
    report += f"\nPDF report saved: `{pdf_path}`\n"

    return report


# --- 3. DOWNLOADABLE-PDF TOOL (in-memory, base64) ---
@mcp.tool()
async def scan_and_publish_download(tickers: list[str]) -> str:
    """Run the scan and return the PDF report as a base64 string (for direct download).

    - Generates the PDF in-memory (works even on read-only file systems).
    - Returns a markdown block containing filename + base64 payload.
    """
    results = _scan_tickers(tickers)
    if not results:
        return "No valid tickers processed - PDF not generated."

    pdf_bytes = generate_pdf_report(results, return_bytes=True)
    b64 = base64.b64encode(pdf_bytes).decode('ascii')
    filename = f"scan_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.pdf"

    md = (
        f"✅ **CLOUD SCAN (download)**\n\n"
        f"PDF filename: `{filename}`\n\n"
        f"```base64\n{b64}\n```"
    )
    return md


if __name__ == "__main__":
    # Changed: Explicitly run with Streamable HTTP transport on port 8000
    # This makes the server accept Streamable-HTTP requests at the same `/sse` URL
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)