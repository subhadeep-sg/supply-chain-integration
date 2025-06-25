# Supply Chain + Income/Population Analytics Integration

## Overview
This is a supply chain forecasting tool that uses data from Freight Analysis Framework, FRED(median household income, 
GDP from retail trade) and US Census Population-particularly for the state of Georgia, and predicts freight value or tons
for selected customer-related commodities.
The tool presently uses a SARIMAX model trained on above-mentioned data between the years 2017-2021 and performs only
**single-step-ahead forecasting**, so it will only predict for **2022** based input data.

## Getting Started
1. Clone repository
    ```
    git clone https://github.com/subhadeep-sg/supplychainintegration.git
    cd supplychainintegration
    ```
2. Create and activate virtual environment
    ```
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```
3. Install Requirements
    ```
    pip install -r requirements.txt
    ```

## Running the API
Start server.
```
uvicorn api.main:app --reload
```
Access locally
Visit http://127.0.0.1:8000 in your browser.

### Stopping API Process
The process can be closed with `Ctrl + C` in the terminal, however if that doesn't work
1. **Check whether the port is still in use** (Windows CMD).
```
netstat -a -n -o | find ":8000"
```
Example output might be:
```
  TCP    127.0.0.1:8000     0.0.0.0:0      LISTENING     12345
```
2. **Kill the process** with:
```
taskkill /PID 12345 /F
```
If this doesn't work, then restart your computer and retry.

## API Endpoints
* GET / - Welcome message.
* GET /options - Lists all valid commodity targets for forecasting.
* POST /forecast - Submit a JSON request to get a forecast.
* GET /version - API version info.

The endpoints were tested using the built-in HTTP client in PyCharm with the 
``test_main.http`` file.

### Example Forecast Request
```
{
  "year": 2022,
  "target_label": "value for Meat/seafood",
  "features": {
    "MEHOINUSGAA672N": 75000,
    "GARETAILNQGSP": 205000,
    "Population": 10700000
  }
}
```
### Example Forecast Response
```
{
  "year": 2022,
  "forecast": 36546.27,
  "target_label": "value_5"      // internal label for a given commodity.
}
```

## Requirements
- fastapi~=0.115.12
- pandas~=2.2.3
- numpy~=2.0.2
- scikit-learn~=1.6.1
- pydantic~=2.11.5
- statsmodels~=0.14.4
- matplotlib~=3.9.4
- plotly~=6.1.2
- uvicorn
- jinja2
- python-multipart
