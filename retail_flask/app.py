from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

import os

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)

# Load model and lookups
project        = joblib.load('retail_forecast_project.joblib')
model          = project['model']
product_lookup = joblib.load('product_lookup.pkl')

# Constants
AVG_DAILY_ORDERS       = 522
WORKING_DAYS_PER_MONTH = 22

# Country encoding map
COUNTRY_MAP = {
    'United Kingdom' : 20, 'Germany'        :  9,
    'France'         :  8, 'EIRE'           :  6,
    'Spain'          : 17, 'Netherlands'    : 12,
    'Belgium'        :  2, 'Switzerland'    : 19,
    'Portugal'       : 16, 'Australia'      :  0,
    'Norway'         : 13, 'Italy'          : 11,
    'Channel Islands':  3, 'Finland'        :  7,
    'Cyprus'         :  4, 'Sweden'         : 18,
    'Austria'        :  1, 'Denmark'        :  5,
    'Poland'         : 15, 'Greece'         : 10,
    'Other'          : 14
}

# Seasonal multipliers
SEASONAL = {
    1 : 0.72, 2 : 0.68, 3 : 0.78,
    4 : 0.75, 5 : 0.79, 6 : 0.76,
    7 : 0.74, 8 : 0.80, 9 : 0.95,
    10: 1.10, 11: 1.35, 12: 1.20
}

def get_season(month):
    if month in [12, 1, 2]: return '❄️ Winter'
    if month in [3, 4, 5]:  return '🌸 Spring'
    if month in [6, 7, 8]:  return '☀️ Summer'
    return '🍂 Autumn'

@app.route('/')
def index():
    products = [
        {'id': k, 'name': v}
        for k, v in sorted(
            product_lookup.items(), key=lambda x: x[1]
        )
    ]
    return render_template(
        'index.html',
        products      = products,
        current_year  = datetime.now().year,
        current_month = datetime.now().month,
        months        = [
            'January', 'February', 'March',
            'April',   'May',      'June',
            'July',    'August',   'September',
            'October', 'November', 'December'
        ]
    )

@app.route('/predict_monthly', methods=['POST'])
def predict_monthly():
    try:
        data  = request.get_json()
        month = int(data['month'])
        year  = int(data['year'])

        input_df = pd.DataFrame([{
            'Description_Encoded': 2000,
            'Country_Encoded'    : 20,
            'Year'               : year,
            'Month'              : month,
            'Day'                : 15,
            'DayOfWeek'          : 2,
            'IsWeekend'          : 0,
            'Quarter'            : (month - 1) // 3 + 1,
            'WeekOfYear'         : min(52, month * 4),
            'Price'              : 3.50
        }])

        avg_qty         = max(0, model.predict(input_df)[0])
        seasonal_factor = SEASONAL.get(month, 1.0)
        monthly_revenue = round(
            avg_qty * 3.50 * AVG_DAILY_ORDERS
            * WORKING_DAYS_PER_MONTH * seasonal_factor, 2
        )

        return jsonify({
            'success'        : True,
            'monthly_revenue': monthly_revenue,
            'month_name'     : datetime(year, month, 1).strftime('%B'),
            'year'           : year,
            'season'         : get_season(month)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_product', methods=['POST'])
def predict_product():
    try:
        data  = request.get_json()
        month = int(data['month'])
        year  = int(data['year'])

        description_encoded = int(data['product_encoded'])
        price               = float(data['price'])
        product_name        = product_lookup.get(
            description_encoded, 'Unknown Product'
        )

        # Predict for each country and sum
        total_predicted_qty = 0

        for country_name, country_code in COUNTRY_MAP.items():
            input_df = pd.DataFrame([{
                'Description_Encoded': description_encoded,
                'Country_Encoded'    : country_code,
                'Year'               : year,
                'Month'              : month,
                'Day'                : 15,
                'DayOfWeek'          : 2,
                'IsWeekend'          : 0,
                'Quarter'            : (month - 1) // 3 + 1,
                'WeekOfYear'         : min(52, month * 4),
                'Price'              : price
            }])

            qty = max(0, model.predict(input_df)[0])
            total_predicted_qty += qty

        # Total monthly product revenue across all countries
        seasonal_factor         = SEASONAL.get(month, 1.0)
        monthly_product_revenue = round(
            total_predicted_qty * price
            * WORKING_DAYS_PER_MONTH * seasonal_factor, 2
        )

        return jsonify({
            'success'                : True,
            'product_name'           : product_name,
            'predicted_qty'          : round(total_predicted_qty, 2),
            'monthly_product_revenue': monthly_product_revenue,
            'month_name'             : datetime(year, month, 1).strftime('%B'),
            'year'                   : year,
            'season'                 : get_season(month)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)  # ← change to True