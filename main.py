import yfinance as yf

result = yf.download("CL=F", start="2024-12-01", end="2025-2-20")

result.to_csv('test.csv', index=True)