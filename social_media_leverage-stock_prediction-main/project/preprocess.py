import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


class EnhancedPreprocessor:
    """Enhanced preprocessing to maximize data"""

    def __init__(self, tweets_csv='data/tweets.csv'):
        self.tweets_csv = tweets_csv
        self.vader = SentimentIntensityAnalyzer()

    def clean_text(self, text):
        """Clean tweet text"""
        if pd.isna(text):
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text.lower()

    def get_vader_sentiment(self, text):
        """Get VADER sentiment"""
        if not text:
            return 0.0
        scores = self.vader.polarity_scores(text)
        return scores['compound']

    def get_textblob_sentiment(self, text):
        """Get TextBlob sentiment"""
        if not text:
            return 0.0
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            return 0.0

    def process(self):
        """Enhanced preprocessing pipeline"""
        print("=" * 80)
        print("ENHANCED PREPROCESSING - MAXIMIZING DATA")
        print("=" * 80)

        # Load
        try:
            df = pd.read_csv(self.tweets_csv)
            print(f"\nLoaded {len(df)} rows")
        except FileNotFoundError:
            print(f"\nError: {self.tweets_csv} not found")
            return None

        # Normalize columns
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        column_mapping = {
            'tweet': 'text',
            'last_price': 'close',
            'px_volume': 'volume',
            '1_day_return': 'return_1d',
            '2_day_return': 'return_2d',
            '3_day_return': 'return_3d',
            '7_day_return': 'return_7d',
            'volatility_10d': 'volatility_10d',
            'volatility_30d': 'volatility_30d',
            'lstm_polarity': 'lstm_sentiment',
            'textblob_polarity': 'textblob_existing'
        }
        df.rename(columns=column_mapping, inplace=True)

        print(f"Columns: {list(df.columns)}")

        # Parse dates
        print("\nProcessing dates...")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Keep reasonable dates
        df = df[(df['date'].dt.year >= 1990) & (df['date'].dt.year <= 2030)]
        df = df.dropna(subset=['date'])

        # For text, fill NaN with empty string instead of dropping
        if 'text' not in df.columns:
            df['text'] = ''
        df['text'] = df['text'].fillna('')

        # Optional stock identifier if present (allows more data across tickers)
        has_stock = 'stock' in df.columns

        # Show which stocks exist in raw data
        if has_stock:
            print("\nStocks present in raw tweets.csv:")
            print(df['stock'].value_counts())

        # Sort for time-series safety
        sort_cols = ['date']
        if has_stock:
            sort_cols = ['stock', 'date']
        df = df.sort_values(sort_cols).reset_index(drop=True)

        print(f"Valid dates: {len(df)} rows")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Clean text
        print("\nCleaning text...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)

        # Compute NEW sentiment (even if existing sentiment is present)
        print("\nComputing sentiment scores...")
        df['vader_sentiment'] = df['cleaned_text'].apply(self.get_vader_sentiment)
        df['textblob_sentiment'] = df['cleaned_text'].apply(self.get_textblob_sentiment)

        # Use existing sentiment if available
        if 'lstm_sentiment' in df.columns:
            df['lstm_sentiment'] = pd.to_numeric(df['lstm_sentiment'], errors='coerce')
        if 'textblob_existing' in df.columns:
            df['textblob_existing'] = pd.to_numeric(df['textblob_existing'], errors='coerce')

        # Classify sentiment
        df['vader_label'] = pd.cut(
            df['vader_sentiment'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['negative', 'neutral', 'positive']
        )

        print(f"\nSentiment distribution:")
        print(df['vader_label'].value_counts())

        # Aggregate by date (and stock if present)
        print("\nAggregating by date...")
        df['date_only'] = df['date'].dt.date
        group_keys = ['date_only']
        if has_stock:
            group_keys = ['stock', 'date_only']

        agg_dict = {
            'vader_sentiment': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'textblob_sentiment': ['mean', 'median', 'std'],
            'cleaned_text': lambda x: ' '.join(x),
        }

        # Add stock columns if they exist
        if 'close' in df.columns:
            agg_dict['close'] = 'first'
        if 'volume' in df.columns:
            agg_dict['volume'] = 'first'
        if 'return_1d' in df.columns:
            agg_dict['return_1d'] = 'first'
        if 'return_2d' in df.columns:
            agg_dict['return_2d'] = 'first'
        if 'return_3d' in df.columns:
            agg_dict['return_3d'] = 'first'
        if 'return_7d' in df.columns:
            agg_dict['return_7d'] = 'first'
        if 'volatility_10d' in df.columns:
            agg_dict['volatility_10d'] = 'first'
        if 'volatility_30d' in df.columns:
            agg_dict['volatility_30d'] = 'first'
        if 'lstm_sentiment' in df.columns:
            agg_dict['lstm_sentiment'] = 'mean'
        if 'textblob_existing' in df.columns:
            agg_dict['textblob_existing'] = 'mean'

        daily = df.groupby(group_keys).agg(agg_dict).reset_index()

        # Compute positive/negative/neutral ratios USING THE SAME GROUP KEYS
        daily = daily.set_index(group_keys)
        total_tweets = daily[('vader_sentiment', 'count')]

        pos_counts = df[df['vader_label'] == 'positive'].groupby(group_keys).size()
        neg_counts = df[df['vader_label'] == 'negative'].groupby(group_keys).size()
        neu_counts = df[df['vader_label'] == 'neutral'].groupby(group_keys).size()

        daily['positive_ratio'] = (pos_counts / total_tweets).fillna(0)
        daily['negative_ratio'] = (neg_counts / total_tweets).fillna(0)
        daily['neutral_ratio'] = (neu_counts / total_tweets).fillna(0)

        # Back to normal columns
        daily = daily.reset_index()

        # Flatten MultiIndex columns
        daily.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) and col[1] else col[0]
            if isinstance(col, tuple) else col
            for col in daily.columns.values
        ]

        # Rename columns for clarity
        daily.rename(columns={'date_only': 'date'}, inplace=True)

        rename_map = {
            'vader_sentiment_mean': 'vader_mean',
            'vader_sentiment_median': 'vader_median',
            'vader_sentiment_std': 'vader_std',
            'vader_sentiment_min': 'vader_min',
            'vader_sentiment_max': 'vader_max',
            'vader_sentiment_count': 'total_tweets',
            'textblob_sentiment_mean': 'textblob_mean',
            'textblob_sentiment_median': 'textblob_median',
            'textblob_sentiment_std': 'textblob_std',
            'cleaned_text_<lambda>': 'combined_text',
            'close_first': 'close',
            'volume_first': 'volume',
            'return_1d_first': 'return_1d',
            'return_2d_first': 'return_2d',
            'return_3d_first': 'return_3d',
            'return_7d_first': 'return_7d',
            'volatility_10d_first': 'volatility_10d',
            'volatility_30d_first': 'volatility_30d',
            'lstm_sentiment_mean': 'lstm_sentiment',
            'textblob_existing_mean': 'textblob_existing'
        }
        daily.rename(columns=rename_map, inplace=True)

        print(f"Aggregated to {len(daily)} rows")

        # Fill NaN for basic stats
        if 'vader_std' in daily.columns:
            daily['vader_std'] = daily['vader_std'].fillna(0)
        if 'textblob_std' in daily.columns:
            daily['textblob_std'] = daily['textblob_std'].fillna(0)

        # Sentiment volatility & range
        daily['sentiment_volatility'] = daily.get('vader_std', 0)
        daily['sentiment_range'] = daily.get('vader_max', 0) - daily.get('vader_min', 0)

        # Sentiment divergence (VADER vs TextBlob)
        if 'vader_mean' in daily.columns and 'textblob_mean' in daily.columns:
            daily['sentiment_divergence'] = (daily['vader_mean'] - daily['textblob_mean']).abs()
        else:
            daily['sentiment_divergence'] = 0.0

        # Create target variable
        if 'return_1d' in daily.columns:
            daily['return_1d'] = pd.to_numeric(daily['return_1d'], errors='coerce')
            daily['target_1d'] = (daily['return_1d'] > 0).astype(int)
            print(f"\nCreated target variable from return_1d")
        else:
            print(f"\nWARNING: No return_1d column found")
            print(f"   Available columns: {list(daily.columns)}")
            if 'close' in daily.columns:
                print("   Attempting to calculate returns from close prices...")
                daily['close'] = pd.to_numeric(daily['close'], errors='coerce')
                daily['return_1d'] = daily['close'].pct_change().shift(-1)
                daily['target_1d'] = (daily['return_1d'] > 0).astype(int)
                print("Created target from close prices")
            else:
                print("   Cannot create target - no price data found")
                return None

        # Sort again for time-series feature engineering
        daily['date'] = pd.to_datetime(daily['date'])
        sort_cols = ['date']
        if has_stock and 'stock' in daily.columns:
            sort_cols = ['stock', 'date']
        daily = daily.sort_values(sort_cols).reset_index(drop=True)

        # Extensive lag features (but not insanely long to avoid losing too many rows)
        print("\nCreating extensive lag features...")
        sentiment_cols = [
            'vader_mean', 'vader_median',
            'positive_ratio', 'negative_ratio',
            'textblob_mean', 'sentiment_volatility'
        ]

        lag_list = [1, 2, 3, 5, 7]

        for col in sentiment_cols:
            if col in daily.columns:
                for lag in lag_list:
                    new_col = f'{col}_lag{lag}'
                    if has_stock and 'stock' in daily.columns:
                        daily[new_col] = (
                            daily.groupby('stock')[col]
                            .shift(lag)
                        )
                    else:
                        daily[new_col] = daily[col].shift(lag)

        # Rolling features (shorter windows to keep more rows)
        rolling_windows = [3, 5, 7]
        for col in ['vader_mean', 'positive_ratio', 'textblob_mean', 'total_tweets']:
            if col in daily.columns:
                for window in rolling_windows:
                    new_col = f'{col}_rolling{window}'
                    if has_stock and 'stock' in daily.columns:
                        daily[new_col] = (
                            daily.groupby('stock')[col]
                            .rolling(window=window, min_periods=1)
                            .mean()
                            .reset_index(level=0, drop=True)
                        )
                    else:
                        daily[new_col] = daily[col].rolling(window=window, min_periods=1).mean()

        # Volume rolling
        if 'volume' in daily.columns:
            for window in rolling_windows:
                new_col = f'volume_rolling{window}'
                if has_stock and 'stock' in daily.columns:
                    daily[new_col] = (
                        daily.groupby('stock')['volume']
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                else:
                    daily[new_col] = daily['volume'].rolling(window=window, min_periods=1).mean()

        # Sentiment momentum
        for period in [3, 5, 7, 14]:
            new_col = f'sentiment_momentum_{period}d'
            if has_stock and 'stock' in daily.columns:
                daily[new_col] = daily.groupby('stock')['vader_mean'].diff(period)
            else:
                daily[new_col] = daily['vader_mean'].diff(period)

        # Price momentum
        if 'close' in daily.columns:
            for period in [3, 5, 7, 14]:
                new_col = f'price_momentum_{period}d'
                if has_stock and 'stock' in daily.columns:
                    daily[new_col] = (
                        daily.groupby('stock')['close']
                        .pct_change(periods=period)
                    )
                else:
                    daily[new_col] = daily['close'].pct_change(periods=period)

        # Volatility of returns (short windows to keep data)
        if 'return_1d' in daily.columns:
            for window in [5, 10]:
                new_col = f'volatility_{window}d'
                if has_stock and 'stock' in daily.columns:
                    daily[new_col] = (
                        daily.groupby('stock')['return_1d']
                        .rolling(window=window)
                        .std()
                        .reset_index(level=0, drop=True)
                    )
                else:
                    daily[new_col] = daily['return_1d'].rolling(window=window).std()

        # Sentiment acceleration (second derivative)
        if has_stock and 'stock' in daily.columns:
            daily['sentiment_acceleration'] = (
                daily.groupby('stock')['vader_mean'].diff().diff()
            )
        else:
            daily['sentiment_acceleration'] = daily['vader_mean'].diff().diff()

        # Tweet volume changes
        if has_stock and 'stock' in daily.columns:
            daily['tweet_volume_change'] = (
                daily.groupby('stock')['total_tweets'].pct_change()
            )
            daily['tweet_volume_change_7d'] = (
                daily.groupby('stock')['total_tweets'].pct_change(periods=7)
            )
        else:
            daily['tweet_volume_change'] = daily['total_tweets'].pct_change()
            daily['tweet_volume_change_7d'] = daily['total_tweets'].pct_change(periods=7)

        # Day of week features
        daily['day_of_week'] = daily['date'].dt.dayofweek
        daily['is_monday'] = (daily['day_of_week'] == 0).astype(int)
        daily['is_friday'] = (daily['day_of_week'] == 4).astype(int)

        # Drop rows with too many NaN (more lenient to keep more data)
        print(f"\nBefore dropping NaN: {len(daily)} rows")
        threshold = len(daily.columns) * 0.6  # 60% of features must be present
        daily = daily.dropna(thresh=threshold)

        # Fill remaining NaN with medians/0
        for col in daily.columns:
            if col in ['date', 'target_1d']:
                continue
            if daily[col].dtype in ['float64', 'int64']:
                if daily[col].isna().sum() > 0:
                    daily[col] = daily[col].fillna(daily[col].median())
            else:
                daily[col] = daily[col].fillna(0)

        # Final check on target
        if 'target_1d' in daily.columns:
            daily = daily.dropna(subset=['target_1d'])

        print(f"After cleaning: {len(daily)} rows")

        # Show stocks in processed dataset
        if has_stock and 'stock' in daily.columns:
            print("\nStocks present in processed dataset:")
            print(daily['stock'].value_counts())

        # Save
        output_file = 'processed_sentiment_stock_data.csv'
        daily.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

        print("\n" + "=" * 80)
        print("ENHANCED PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"\nFinal dataset:")
        print(f"  Shape: {daily.shape}")
        print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
        print(f"  Total features: {len(daily.columns)}")

        if 'target_1d' in daily.columns:
            up_pct = daily['target_1d'].mean() * 100
            print(f"\nTarget distribution:")
            print(f"  Up days:   {daily['target_1d'].sum()} ({up_pct:.1f}%)")
            print(f"  Down days: {len(daily) - daily['target_1d'].sum()} ({100 - up_pct:.1f}%)")

        return daily


def main():
    preprocessor = EnhancedPreprocessor(tweets_csv='data/tweets.csv')
    df = preprocessor.process()

    if df is not None:
        print("run python evaluate.py")
    else:
        print("\n Preprocessing failed.")


if __name__ == '__main__':
    main()
