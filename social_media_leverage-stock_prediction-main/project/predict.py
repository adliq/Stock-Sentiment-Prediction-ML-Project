import pandas as pd
import numpy as np
import pickle

def load_model(prefix='best_model'):
    """Load trained model"""
    try:
        with open(f'{prefix}.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{prefix}_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{prefix}_selector.pkl', 'rb') as f:
            selector = pickle.load(f)
        with open(f'{prefix}_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, scaler, selector, metadata
    except FileNotFoundError:
        print("Error: Model files not found. Run training script first")
        return None, None, None, None


def predict_labels(X, model, scaler, selector, threshold):
    """
    Make label predictions using the stored threshold.
    Probabilities are used internally but NOT returned or stored.
    """
    X_selected = selector.transform(X)
    X_scaled = scaler.transform(X_selected)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds


def main():
    print("=" * 80)
    print("STOCK PREDICTION - INFERENCE")
    print("=" * 80)

    # Load model
    model, scaler, selector, metadata = load_model('best_model')
    if model is None:
        return

    print("\nModel loaded")

    # Handle both old and new model formats
    if 'model_type' in metadata:
        print(f"  Type: {metadata['model_type']}")
    elif 'use_ensemble' in metadata:
        model_type = "Ensemble" if metadata['use_ensemble'] else "GradientBoosting (Regularized)"
        print(f"  Type: {model_type}")
    else:
        print("  Type: Unknown")

    print(f"  Total features: {len(metadata['feature_names'])}")
    print(f"  Selected features: {metadata['n_features']}")
    print(f"  Top 5 features: {metadata['selected_features'][:5]}")
    print(f"  Threshold: {metadata['best_threshold']:.3f}")

    # Load data
    try:
        df = pd.read_csv('processed_sentiment_stock_data.csv')
        print(f"\nData loaded: {len(df)} rows")
    except FileNotFoundError:
        print("\nprocessed_sentiment_stock_data.csv not found")
        return

    # Ensure proper time ordering
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        sort_cols = ['date']
        if 'stock' in df.columns:
            sort_cols = ['stock', 'date']
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Show which stocks are present for inference
    if 'stock' in df.columns:
        print("\nStocks present in inference dataset:")
        print(df['stock'].value_counts())

    # Create meta features if needed
    print("\nCreating meta features if needed...")

    if 'day_of_week' in df.columns and 'price_momentum_3d' in df.columns:
        if 'day_momentum_interaction' not in df.columns:
            df['day_momentum_interaction'] = df['day_of_week'] * df['price_momentum_3d']

    if 'vader_mean' in df.columns and 'price_momentum_3d' in df.columns:
        if 'sentiment_price_interaction' not in df.columns:
            df['sentiment_price_interaction'] = df['vader_mean'] * df['price_momentum_3d']

    momentum_cols = [c for c in df.columns if 'price_momentum' in c]
    if len(momentum_cols) >= 2:
        if 'momentum_strength' not in df.columns:
            df['momentum_strength'] = df[momentum_cols].mean(axis=1)
        if 'momentum_consistency' not in df.columns:
            df['momentum_consistency'] = df[momentum_cols].std(axis=1)

    sentiment_cols = [c for c in df.columns if 'vader_mean' in c and 'lag' in c]
    if len(sentiment_cols) >= 2:
        if 'sentiment_consistency' not in df.columns:
            df['sentiment_consistency'] = df[sentiment_cols].std(axis=1)

    if 'tweet_volume_change' in df.columns and 'total_tweets' in df.columns:
        if 'volume_acceleration' not in df.columns:
            df['volume_acceleration'] = df['tweet_volume_change'] * df['total_tweets']

    print("Meta features ready")

    # Check required features
    missing_features = [f for f in metadata['feature_names'] if f not in df.columns]
    if missing_features:
        print(f"\nMissing {len(missing_features)} features:")
        for feat in missing_features[:10]:
            print(f"  - {feat}")
        print("\nUsing available features only...")

        available_features = [f for f in metadata['feature_names'] if f in df.columns]
        X = df[available_features].values
        print("\nSome features missing - predictions may be less accurate")
    else:
        X = df[metadata['feature_names']].values

    # Predict labels
    preds = predict_labels(X, model, scaler, selector, metadata['best_threshold'])
    df['predicted_up'] = preds

    print("\n" + "=" * 80)
    print("OVERALL PREDICTION BALANCE")
    print("=" * 80)

    if 'target_1d' in df.columns:
        print("\nPredictions:")
        print(df['predicted_up'].value_counts().sort_index())
        print(f"\nPredicted UP rate:  {df['predicted_up'].mean():.1%}")
        print(f"Actual UP rate:     {df['target_1d'].mean():.1%}")

        from sklearn.metrics import (
            confusion_matrix,
            precision_score,
            recall_score,
            f1_score,
        )

        overall_acc = (df['predicted_up'] == df['target_1d']).mean()
        print(f"\nOVERALL ACCURACY: {overall_acc:.1%}")

        cm = confusion_matrix(df['target_1d'], df['predicted_up'])
        print(f"\nConfusion Matrix:")
        print(f"                 Pred DOWN   Pred UP")
        print(f"Actual DOWN:     {cm[0][0]:8d}   {cm[0][1]:8d}")
        print(f"Actual UP:       {cm[1][0]:8d}   {cm[1][1]:8d}")

        precision = precision_score(df['target_1d'], df['predicted_up'], zero_division=0)
        recall = recall_score(df['target_1d'], df['predicted_up'], zero_division=0)
        f1 = f1_score(df['target_1d'], df['predicted_up'], zero_division=0)

        print(f"\nClassification Metrics:")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall:    {recall:.1%}")
        print(f"  F1 Score:  {f1:.3f}")

    # Last 30 predictions
    print("\n" + "=" * 80)
    print("LAST 30 PREDICTIONS (Chronological)")
    print("=" * 80)

    show_cols = ['date']
    if 'stock' in df.columns:
        show_cols.append('stock')
    if 'return_1d' in df.columns:
        show_cols.append('return_1d')
    if 'target_1d' in df.columns:
        show_cols.append('target_1d')
    show_cols.append('predicted_up')

    recent = df.tail(30)[show_cols].copy()

    if 'target_1d' in recent.columns:
        recent['correct'] = (recent['target_1d'] == recent['predicted_up']).astype(int)

    recent['direction'] = recent['predicted_up'].map({0: "DOWN", 1: "UP"})

    for _, row in recent.iterrows():
        date_str = str(row['date'])[:10] if 'date' in row else 'N/A'
        stock_str = row['stock'] if 'stock' in row else 'N/A'
        pred_str = row['direction']

        if 'target_1d' in row and pd.notna(row['target_1d']):
            actual_str = 'UP' if row['target_1d'] == 1 else 'DOWN'
            correct_str = "OK" if row.get('correct', 0) == 1 else "X"
            print(f"{date_str} | Stock: {stock_str} | Predicted: {pred_str} | Actual: {actual_str} {correct_str}")
        else:
            print(f"{date_str} | Stock: {stock_str} | Predicted: {pred_str}")

    if 'correct' in recent.columns:
        print(f"\nLast 30 predictions accuracy: {recent['correct'].mean():.1%}")

    # Current prediction
    print("\n" + "=" * 80)
    print("CURRENT PREDICTION")
    print("=" * 80)

    last = df.iloc[-1]
    pred_direction = "UP" if last['predicted_up'] == 1 else "DOWN"
    stock_str = last['stock'] if 'stock' in last else 'N/A'

    print(f"\nDate: {last['date']}")
    print(f"Stock: {stock_str}")
    print(f"Prediction: {pred_direction}")

    # Save predictions
    save_cols = ['date', 'predicted_up']
    if 'stock' in df.columns:
        save_cols.insert(1, 'stock')
    if 'return_1d' in df.columns:
        save_cols.insert(1, 'return_1d')
    if 'target_1d' in df.columns:
        save_cols.insert(2, 'target_1d')

    existing_cols = [c for c in save_cols if c in df.columns]
    df[existing_cols].to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")


if __name__ == '__main__':
    main()
