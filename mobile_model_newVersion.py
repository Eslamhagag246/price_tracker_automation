
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from datetime import timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_and_preprocess_data(filepath=r'C:\projects\final_project\full_price_prediction\mobile_cleaned_70K.csv'):

    df = pd.read_csv(filepath)

    df['price'] = df['price'].astype(str)
    df['price'] = df['price'].str.replace('EGP', '', regex=False)
    df['price'] = df['price'].str.replace(',', '', regex=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['date'] = pd.to_datetime(df['date'])

    df['product_key'] = (
        df['name'].str.lower().str.strip() + ' ' +
        df['website'].str.lower() + ' ' +
        df['ram_gb'].astype(str) + ' ' +
        df['storage_gb'].astype(str)
    )

    df_daily = df.groupby(['product_key', 'date']).agg({
        'price': 'mean',
        'name': 'first',
        'brand': 'first',
        'website': 'first',
        'ram_gb': 'first',
        'storage_gb': 'first',
        'URL': 'last',
        'timestamp': 'first'
    }).reset_index()

    df_daily = df_daily.sort_values(['product_key', 'date'])

    return df_daily


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def engineer_features(pdf):

    pdf = pdf.sort_values('date').copy()

    pdf['day_index'] = (pdf['date'] - pdf['date'].min()).dt.days
    pdf['dayofweek'] = pdf['date'].dt.dayofweek
    pdf['day_of_month'] = pdf['date'].dt.day
    pdf['month'] = pdf['date'].dt.month

    pdf['rolling_avg_3'] = pdf['price'].rolling(3, min_periods=1).mean()
    pdf['rolling_avg_7'] = pdf['price'].rolling(7, min_periods=1).mean()
    pdf['rolling_std_3'] = pdf['price'].rolling(3, min_periods=1).std().fillna(0)

    pdf['price_lag_1'] = pdf['price'].shift(1).fillna(pdf['price'].iloc[0])
    pdf['price_lag_3'] = pdf['price'].shift(3).fillna(pdf['price'].iloc[0])
    pdf['price_lag_7'] = pdf['price'].shift(7).fillna(pdf['price'].iloc[0])

    pdf['pct_change_1'] = pdf['price'].pct_change().fillna(0)
    pdf['pct_change_3'] = pdf['price'].pct_change(3).fillna(0)

    pdf['ram_normalized'] = pdf['ram_gb'] / 16.0
    pdf['storage_normalized'] = pdf['storage_gb'] / 1024.0
    pdf['specs_score'] = (pdf['ram_gb'] / 4.0) + (pdf['storage_gb'] / 128.0)

    return pdf


# ═══════════════════════════════════════════════════════════
# GLOBAL MODEL TRAINING WITH EVALUATION
# ═══════════════════════════════════════════════════════════

FEATURE_COLS = [
    'day_index', 'dayofweek', 'day_of_month', 'month',
    'rolling_avg_3', 'rolling_avg_7', 'rolling_std_3',
    'price_lag_1', 'price_lag_3', 'price_lag_7',
    'pct_change_1', 'pct_change_3',
    'ram_normalized', 'storage_normalized', 'specs_score'
]

MODEL_PATH = "mobile_price_model.pkl"


def train_global_model(filepath, min_obs=10, test_size=0.2):

    df = load_and_preprocess_data(filepath)

    X_all = []
    y_all = []

    for product_key in df['product_key'].unique():

        pdf = df[df['product_key'] == product_key]

        if len(pdf) < min_obs:
            continue

        pdf = engineer_features(pdf)

        X_all.append(pdf[FEATURE_COLS])
        y_all.append(pdf['price'])

    X_all = pd.concat(X_all)
    y_all = pd.concat(y_all)

    print(f"\nTraining global mobile model on {len(X_all)} samples")

    # Time-based split
    split_idx = int(len(X_all) * (1 - test_size))

    X_train = X_all.iloc[:split_idx]
    X_test = X_all.iloc[split_idx:]

    y_train = y_all.iloc[:split_idx]
    y_test = y_all.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE")
    print("="*60)

    print("\nTRAINING PERFORMANCE")
    print(f"MAE:  {train_mae:,.2f} EGP")
    print(f"R²:   {train_r2:.4f}")
    print(f"RMSE: {train_rmse:,.2f} EGP")

    print("\nTEST PERFORMANCE")
    print(f"MAE:  {test_mae:,.2f} EGP")
    print(f"R²:   {test_r2:.4f}")
    print(f"RMSE: {test_rmse:,.2f} EGP")

    return model


def save_global_model(model):

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")


def load_global_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found")

    return joblib.load(MODEL_PATH)


# ═══════════════════════════════════════════════════════════
# FORECASTING
# ═══════════════════════════════════════════════════════════

def forecast_product(pdf, days_ahead=7, model=None):

    pdf = engineer_features(pdf)

    X = pdf[FEATURE_COLS]
    y = pdf['price']

    if model is None:
        model = load_global_model()

    history_prices = list(pdf['price'].values)

    last_date = pdf['date'].iloc[-1]
    last_day_index = pdf['day_index'].iloc[-1]

    forecasts = []

    for i in range(days_ahead):

        future_date = last_date + timedelta(days=i+1)

        price_lag_1 = history_prices[-1]
        price_lag_3 = history_prices[-3] if len(history_prices) >= 3 else history_prices[0]
        price_lag_7 = history_prices[-7] if len(history_prices) >= 7 else history_prices[0]

        rolling_avg_3 = np.mean(history_prices[-3:])
        rolling_avg_7 = np.mean(history_prices[-7:])
        rolling_std_3 = np.std(history_prices[-3:])

        row = [[
            last_day_index+i+1,
            future_date.dayofweek,
            future_date.day,
            future_date.month,
            rolling_avg_3,
            rolling_avg_7,
            rolling_std_3,
            price_lag_1,
            price_lag_3,
            price_lag_7,
            0,
            0,
            pdf['ram_normalized'].iloc[-1],
            pdf['storage_normalized'].iloc[-1],
            pdf['specs_score'].iloc[-1]
        ]]

        pred = model.predict(row)[0]

        forecasts.append(pred)
        history_prices.append(pred)

    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    n = len(pdf)

    if n >= 30:
        confidence = "High"
    elif n >= 15:
        confidence = "Medium"
    else:
        confidence = "Low"

    last_price = float(pdf['price'].iloc[-1])
    future_price = forecasts[-1]
    trend_pct = ((future_price - last_price) / last_price) * 100

    if trend_pct < -3:
        signal = "buy"
        signal_text = "💰 Buy Opportunity"
        signal_desc = "Price expected to drop"
    elif trend_pct > 3:
        signal = "wait"
        signal_text = "⏳ Wait Before Buying"
        signal_desc = "Price expected to rise"
    else:
        signal = "neutral"
        signal_text = "📊 Stable Price"
        signal_desc = "Price expected to stay stable"

    return {
        'pdf': pdf,
        'forecast_dates': forecast_dates,
        'forecast_prices': np.array(forecasts),
        'mae': mae,
        'r2': r2,
        'last_price': last_price,
        'avg_price': float(pdf['price'].mean()),
        'min_price': float(pdf['price'].min()),
        'max_price': float(pdf['price'].max()),
        'n_obs': n,
        'confidence': confidence,
        'signal': signal,
        'signal_text': signal_text,
        'signal_desc': signal_desc,
        'trend_pct': trend_pct,
        'model_type': 'Global Linear Regression'
    }


# ═══════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*70)
    print("🚀 TRAINING GLOBAL MOBILE PRICE MODEL")
    print("="*70)

    filepath = r"C:\projects\final_project\full_price_prediction\mobile_cleaned_70K.csv"

    model = train_global_model(filepath)

    save_global_model(model)

    print("\n✅ Training complete")