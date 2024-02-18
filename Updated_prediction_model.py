import time                                                         # CRYPTO PRICE PREDICTOR MODEL
import datetime
import numpy as np                                                  # BETA          -- MAY PROVIDE WRONG PREDICTION ---
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob 
from binance.client import Client
import requests  
from newsapi import NewsApiClient  

# Function to fetch historical prices from Binance
def get_historical_prices(symbol, interval, limit):
    # Implement Binance API data fetching here
    # Replace the placeholders with your Binance API key and secret
    api_key = 'KOwrEeX3F4N5462Pro7gBaokT8eYCX8lCRYZm7k7ZWBQtaaRpOPBxmfunOkbbqlQ'
    api_secret = 'vSxmf48SHc1jnpd8DjPxjkjY2KnUgDZBkyOwkaSqcnyiPpENinKEclqkUcsA9uc7'
    client = Client(api_key, api_secret)

    # Fetch historical price data using the Binance API
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    # We only need the first 5 columns
    df = df[['timestamp', 'open', 'high', 'low', 'close']]

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Function to fetch news data and perform sentiment analysis
def fetch_news_sentiment(api_key, keyword):
    url = f'https://newsapi.org/v2/everything?q={keyword}&sortBy=popularity,publishedAt&apiKey={api_key}&language=en'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        sentiment_scores = []
        popular_news = []
        latest_news = []

        for article in articles:
            sentiment_score = analyze_sentiment(article['title'])
            sentiment_scores.append(sentiment_score)

            if 'publishedAt' in article and article['publishedAt']:
                news_date = datetime.datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                news_date_str = news_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                news_date_str = "N/A"

            news_item = {
                'title': article['title'],
                'source': article['source']['name'],
                'published_at': news_date_str
            }

            try:
                if article['sortBysAvailable']:
                    popular_news.append(news_item)
                else:
                    latest_news.append(news_item)
            except KeyError:
                latest_news.append(news_item)

        mean_sentiment_score = np.mean(sentiment_scores)
        return mean_sentiment_score, popular_news, latest_news
    else:
        print("Error fetching news data.")
        return None, [], []

# Function to perform sentiment analysis on the news title
def analyze_sentiment(title):
    blob = TextBlob(title)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to scale data using MinMaxScaler
def scale_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

# Function to prepare data for LSTM input
def prepare_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Function to predict the next day's price using the trained LSTM model
def predict_next_day_price(model, data, scaler, window_size):
    last_window = data[-window_size:]
    last_window = np.reshape(last_window, (1, window_size, 1))
    next_day_price = model.predict(last_window)
    next_day_price = scaler.inverse_transform(next_day_price)[0][0]
    return next_day_price

# Function to calculate suggested leverage based on predicted price change and user investment
def calculate_suggested_leverage(current_price, predicted_price, investment):
    # Calculate the percentage change in price
    price_change = abs((predicted_price - current_price) / current_price)

    # Determine the suggested leverage based on the predicted change and user investment
    if price_change <= 0.005:
        suggested_leverage = 1  # Low leverage for small predicted changes
    elif price_change <= 0.01:
        suggested_leverage = 2  # Medium leverage for moderate predicted changes
    else:
        # Calculate leverage based on user's investment
        suggested_leverage = int(5 * (investment / current_price))

    return suggested_leverage

# Function to check the exit strategy (Stop-loss of 2%)
def check_exit_strategy(current_price, predicted_price, stop_loss=0.02):
    # Calculate the percentage change in price
    price_change = abs((current_price - predicted_price) / current_price)

    if price_change >= stop_loss:
        return True
    else:
        return False

# Function to implement moving average crossover strategy
def moving_average_crossover_strategy(df, short_window, long_window):
    signals = pd.DataFrame(index=df.index)
    signals['Signal'] = 0.0
    signals['Short_MA'] = df['close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long_MA'] = df['close'].rolling(window=long_window, min_periods=1, center=False).mean()

    signals.iloc[short_window:, signals.columns.get_loc('Signal')] = np.where(
        signals['Short_MA'].iloc[short_window:] > signals['Long_MA'].iloc[short_window:], 1.0, 0.0
    )

    signals['Position'] = signals['Signal'].diff()
    return signals

# Function to analyze order book and volume and return the predicted price change
def calculate_rsi(df, window=14):
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate the Moving Average Convergence Divergence (MACD)
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return macd, signal

# Function to analyze order book and volume and return the predicted price change
def analyze_order_book_and_volume(order_book, volume):
    # Convert order_book and volume to one-dimensional arrays (if they are not already)
    order_book = np.array(order_book).flatten()
    volume = np.array(volume).flatten()

    # Calculate RSI
    delta = np.diff(order_book)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate MACD
    macd, signal = calculate_macd(pd.DataFrame({'close': order_book}))

    # Implement your strategy here
    # For demonstration purposes, let's assume a simple strategy
    if rsi > 70 and macd.iloc[-1] > signal.iloc[-1]:
        predicted_price_change = 0.005  # Predict a 0.5% price increase
    elif rsi < 30 and macd.iloc[-1] < signal.iloc[-1]:
        predicted_price_change = -0.005  # Predict a 0.5% price decrease
    else:
        predicted_price_change = 0.0  # No significant price change predicted

    return predicted_price_change
# Main loop
if __name__ == "__main__":
    currency_pair = input("Enter the currency pair (e.g., BTCUSDT): ").upper()
    interval = Client.KLINE_INTERVAL_1DAY
    window_size = 10
    short_window = 5
    long_window = 10
    prediction_days = 1
    limit = 30 * window_size  # Fetch 30 days' worth of data

    # Ask user for investment amount
    investment = float(input("Enter your investment amount (in USD): "))

    # Replace
    news_api_key = '83d87fe83bd447c7a4421b0d85ebdd64'
    while True:
        df = get_historical_prices(currency_pair, interval, limit)
        data = df['close'].values.reshape(-1, 1)

        scaler, scaled_data = scale_data(data)
        X, y = prepare_data(scaled_data, window_size)

        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=50, batch_size=16, verbose=0)

        next_day_price_lstm = predict_next_day_price(model, scaled_data, scaler, window_size)

        # Sample order book and volume data (replace with actual data)
        order_book_data = np.array([5000, 5020, 5015, 5022, 5018, 5015, 5020, 5025, 5010, 5028])
        volume_data = np.array([10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000])

        # Analyze order book and volume to predict price change
        predicted_price_change_obv = analyze_order_book_and_volume(order_book_data, volume_data)
        estimated_price_obv = next_day_price_lstm * (1 + predicted_price_change_obv)

        current_price = float(df['close'].iloc[-1])
        last_data = df['close'].tail(window_size).values.reshape(-1, 1)
        scaled_last_data = scaler.transform(last_data)
        next_day_price_lstm = predict_next_day_price(model, scaled_last_data, scaler, window_size)

        df = moving_average_crossover_strategy(df, short_window, long_window)
        signal = df['Signal'].iloc[-1]

        if signal == 1:
            trade_signal = "Long"
        else:
            trade_signal = "Short"

        # Fetch news data and perform sentiment analysis
        keyword = currency_pair.split('USDT')[0]  # Extract the base currency from the currency pair
        news_sentiment, popular_news, latest_news = fetch_news_sentiment(news_api_key, keyword)

        # Predict next day's price based on news sentiment
        estimated_price_news = next_day_price_lstm * (1 + news_sentiment)

        # Analyze order book and volume to predict price change
        predicted_price_change_obv = analyze_order_book_and_volume(order_book_data, volume_data)
        estimated_price_obv = next_day_price_lstm * (1 + predicted_price_change_obv)

        suggested_leverage = calculate_suggested_leverage(current_price, estimated_price_news, investment)
        entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Current {currency_pair} Price: {round(current_price, 4)} USDT")
        print(f"Next Day's Predicted {currency_pair} Price (LSTM): {round(next_day_price_lstm, 4)} USDT")
        print(f"Next Day's Predicted {currency_pair} Price (OBV): {round(estimated_price_obv, 4)} USDT")
        print(f"Estimated Next Day's Price (Based on News Sentiment): {round(estimated_price_news, 4)} USDT")
        print(f"Trade Signal: {trade_signal}")
        print(f"Suggested Leverage: {suggested_leverage}")
        print(f"Time of Entry: {entry_time}")

        '''
        # Display popular and latest news
        print("\nPopular News:")
       for i, news_item in enumerate(popular_news, start=1):
            print(f"{i}. {news_item['title']} - {news_item['source']} ({news_item['published_at']})")

        print("\nLatest News:")
        for i, news_item in enumerate(latest_news, start=1):
            print(f"{i}. {news_item['title']} - {news_item['source']} ({news_item['published_at']})")
        '''

        # Check exit strategy
        if check_exit_strategy(current_price, estimated_price_news):
            print("Exit Trade: Stop-loss triggered!")

        time.sleep(60)  # Wait for 1 minute before predicting the next day's price again
