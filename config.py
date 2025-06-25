commodity_mapping = {5: 'Meat/seafood', 8: 'Alcoholic beverages',
                     9: 'Tobacco prods.', 21: 'Pharmaceuticals'}

predictive_features = ['MEHOINUSGAA672N', 'GARETAILNQGSP', 'Population']

feature_units = {'value': 'Million Dollars', 'tons': 'Million Tons',
                 'Value': 'Million Dollars', 'Tons': 'Million Tons'}

forecast_options = {}

for code, name in commodity_mapping.items():
    forecast_options[f"value for {name}"] = f"value_{code}"
    forecast_options[f"tons for {name}"] = f"tons_{code}"

unit_conversion = {
    'MEHOINUSGAA672N': 1e-6,    # Converted Dollars to a million Dollars
    'GARETAILNQGSP': 1,         # in millions already
    'Population': 1             # using absolute quantity
}