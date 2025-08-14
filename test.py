import requests

url = "http://127.0.0.1:8000/predict"  # Change this if running elsewhere

SEQUENCE_LENGTH = 20

example_day = {
    "close": 800.5,
    "volume": 123456,
    "RSI_14": 56.2,
    "DMA_20": 790.1,
    "DMA_50": 780.5,
    "DMA_100": 770.9,
    "SUPPORT_20": 785.0,
    "RESIST_20": 810.0,
    "PE": 25.5,
    "PB": 4.1
}

data = {
    "stock_name": "TATAMOTORS",
    "features": [example_day for _ in range(SEQUENCE_LENGTH)]  # Repeat for 20 days
}

response = requests.post(url, json=data)
print("Status code:", response.status_code)
print("Response:", response.json())
