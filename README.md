Wall Street Publisher (MCP server)

Features added:

- Scan stocks and compute DCF-based fair values (existing)
- PDF report generation with detailed per-ticker DCF insights (new)

Usage

1. Start dev server:
   - uv run fastmcp dev main.py
   - If port conflict occurs, stop any running dev servers or change proxy port.

2. Trigger scan (via your MCP client):
   - Use the `scan_and_publish` tool with a list of tickers (e.g. ['AAPL','MSFT']).
   - The tool returns a markdown summary and saves a PDF under `reports/` (filename printed in the response).

Output

- Quick markdown table in the chat (immediate)
- Detailed PDF at `reports/scan_<timestamp>.pdf` containing assumptions, 5-year FCF projections, PVs, terminal value, and headline metrics.

Notes

- Dependencies: `fpdf2` was added for PDF generation.
- The PDF generation is synchronous and lightweight; for heavy workloads you can offload it to a thread or async worker.
