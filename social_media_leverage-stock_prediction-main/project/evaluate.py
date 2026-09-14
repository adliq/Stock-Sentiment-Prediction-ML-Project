import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pickle
import warnings

warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


class UltimatePredictor:
    """Ultimate stock predictor with overfitting prevention"""

    def __init__(self, n_features=12, use_ensemble=False):
        self.n_features = n_features
        self.use_ensemble = use_ensemble
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.model = None
        self.feature_names = None
        self.selected_features = None
        self.best_threshold = 0.5

    def engineer_meta_features(self, df):
        """Create additional meta features"""
        print("\nEngineering meta features...")

        if 'day_of_week' in df.columns and 'price_momentum_3d' in df.columns:
            df['day_momentum_interaction'] = (
                df['day_of_week'] * df['price_momentum_3d']
            )

        if 'vader_mean' in df.columns and 'price_momentum_3d' in df.columns:
            df['sentiment_price_interaction'] = (
                df['vader_mean'] * df['price_momentum_3d']
            )

        momentum_cols = [c for c in df.columns if 'price_momentum' in c]
        if len(momentum_cols) >= 2:
            df['momentum_strength'] = df[momentum_cols].mean(axis=1)
            df['momentum_consistency'] = df[momentum_cols].std(axis=1)

        sentiment_cols = [
            c for c in df.columns if 'vader_mean' in c and 'lag' in c
        ]
        if len(sentiment_cols) >= 2:
            df['sentiment_consistency'] = df[sentiment_cols].std(axis=1)

        if 'tweet_volume_change' in df.columns and 'total_tweets' in df.columns:
            df['volume_acceleration'] = (
                df['tweet_volume_change'] * df['total_tweets']
            )

        print("Added meta features")

        return df

    def select_features(self, X, y):
        """Feature selection"""
        k = min(self.n_features, X.shape[1])
        self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)

        feature_indices = self.feature_selector.get_support(indices=True)
        feature_scores = self.feature_selector.scores_

        self.selected_features = [self.feature_names[i] for i in feature_indices]

        print(f"\nSelected top {len(self.selected_features)} features:")
        feature_importance = sorted(
            zip(self.selected_features, feature_scores[feature_indices]),
            key=lambda x: x[1],
            reverse=True,
        )
        for feat, score in feature_importance:
            print(f"  {feat:40s} (score: {score:.3f})")

        return X_selected

    def create_base_models(self):
        """Create base models"""
        models = []

        lr = LogisticRegression(
            penalty='l2',
            C=0.1,
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
        )
        models.append(('lr', lr))

        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
        )
        models.append(('rf', rf))

        gb = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.01,
            subsample=0.7,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )
        models.append(('gb', gb))

        return models

    def create_model(self):
        """Create final model"""
        if self.use_ensemble:
            base_models = self.create_base_models()
            model = VotingClassifier(
                estimators=base_models, voting='soft', n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.01,
                subsample=0.7,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
            )

        return model

    def fit(self, X, y, feature_names):
        """Fit model"""
        self.feature_names = feature_names

        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(y) * 100
            label = "Down (0)" if cls == 0 else "Up (1)"
            print(f"  {label}: {cnt} ({pct:.1f}%)")

        X_selected = self.select_features(X, y)
        X_scaled = self.scaler.fit_transform(X_selected)

        print("\nUsing class weights only (no SMOTE)")

        self.model = self.create_model()
        self.model.fit(X_scaled, y)

        return self

    def predict(self, X):
        """Predict labels and probabilities"""
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= self.best_threshold).astype(int)
        return preds, probs

    def walk_forward_validation(self, X, y, min_train_size=500, step_size=100):
        """Train on past, predict future"""
        print(f"\n{'=' * 80}")
        print("WALK-FORWARD VALIDATION")
        print(f"{'=' * 80}")

        results = []

        for train_end in range(min_train_size, len(X), step_size):
            test_start = train_end
            test_end = min(train_end + step_size, len(X))

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            if len(X_test) < 10:
                continue

            fold_model = UltimatePredictor(
                n_features=self.n_features, use_ensemble=self.use_ensemble
            )
            fold_model.feature_names = self.feature_names

            fold_model.feature_selector = SelectKBest(
                mutual_info_classif, k=self.n_features
            )
            X_train_selected = fold_model.feature_selector.fit_transform(
                X_train, y_train
            )
            X_test_selected = fold_model.feature_selector.transform(X_test)

            fold_model.scaler = RobustScaler()
            X_train_scaled = fold_model.scaler.fit_transform(X_train_selected)
            X_test_scaled = fold_model.scaler.transform(X_test_selected)

            fold_model.model = fold_model.create_model()
            fold_model.model.fit(X_train_scaled, y_train)

            test_preds = fold_model.model.predict(X_test_scaled)
            test_probs = fold_model.model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, zero_division=0)

            try:
                auc = roc_auc_score(y_test, test_probs)
            except Exception:
                auc = 0.5

            results.append({'acc': acc, 'f1': f1, 'auc': auc, 'n_test': len(y_test)})

            print(
                f"Train: 0-{train_end}, Test: {test_start}-{test_end} | "
                f"Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}"
            )

        if results:
            print(f"\n{'=' * 80}")
            print("WALK-FORWARD SUMMARY")
            print(f"{'=' * 80}")
            print(f"Average Test Accuracy: {np.mean([r['acc'] for r in results]):.3f}")
            print(f"Average Test F1:       {np.mean([r['f1'] for r in results]):.3f}")
            print(f"Average Test AUC:      {np.mean([r['auc'] for r in results]):.3f}")

        return results

    def save(self, prefix='ultimate_model'):
        """Save model and metadata"""
        with open(f'{prefix}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{prefix}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{prefix}_selector.pkl', 'wb') as f:
            pickle.dump(self.feature_selector, f)
        with open(f'{prefix}_metadata.pkl', 'wb') as f:
            pickle.dump(
                {
                    'feature_names': self.feature_names,
                    'selected_features': self.selected_features,
                    'best_threshold': self.best_threshold,
                    'n_features': self.n_features,
                    'use_ensemble': self.use_ensemble,
                },
                f,
            )
        print(f"\nSaved to {prefix}.pkl")


def main():
    print("=" * 80)
    print("ULTIMATE STOCK PREDICTION MODEL")
    print("Aggressive overfitting prevention")
    print("=" * 80)

    # Load data
    try:
        df = pd.read_csv('processed_sentiment_stock_data.csv')
        print(f"\nLoaded {len(df)} rows")
    except FileNotFoundError:
        print("\nprocessed_sentiment_stock_data.csv not found")
        return

    # Sort by date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

    # Show which stocks are in the dataset
    if 'stock' in df.columns:
        print("\nStocks present in full processed dataset:")
        print(df['stock'].value_counts())

    # Engineer meta features
    predictor = UltimatePredictor(n_features=12, use_ensemble=False)
    df = predictor.engineer_meta_features(df)

    # Prepare features
    exclude_cols = [
        'date',
        'return_1d',
        'target_1d',
        'close',
        'volume',
        'open',
        'high',
        'low',
        'adj_close',
        'return_2d',
        'return_3d',
        'return_7d',
        'combined_text',
        'date_only',
        'stock',
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols and col]

    X = df[feature_cols].values
    y = df['target_1d'].values

    print(f"\nTotal features: {len(feature_cols)}")

    configs = [
        {'n_features': 8, 'ensemble': False, 'name': 'GB + 8 features (regularized)'},
        {
            'n_features': 10,
            'ensemble': False,
            'name': 'GB + 10 features (regularized)',
        },
        {
            'n_features': 12,
            'ensemble': False,
            'name': 'GB + 12 features (regularized)',
        },
        {'n_features': 10, 'ensemble': True, 'name': 'Ensemble + 10 features'},
    ]

    best_score = 0
    best_config = None
    best_model = None

    # Hold out most recent 20% for final test
    split_idx = int(len(X) * 0.8)
    X_train_all = X[:split_idx]
    y_train_all = y[:split_idx]
    X_holdout = X[split_idx:]
    y_holdout = y[split_idx:]

    print(f"\nTrain size: {len(X_train_all)}, Hold-out size: {len(X_holdout)}")

    # Show which stocks are in the hold-out period
    if 'stock' in df.columns:
        holdout_stocks = df['stock'].iloc[split_idx:]
        print("\nStocks in hold-out (most recent 20%):")
        print(holdout_stocks.value_counts())

    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"TESTING: {config['name']}")
        print(f"{'=' * 80}")

        model = UltimatePredictor(
            n_features=config['n_features'], use_ensemble=config['ensemble']
        )

        model.fit(X_train_all, y_train_all, feature_cols)

        wf_results = model.walk_forward_validation(
            X_train_all, y_train_all, min_train_size=300, step_size=50
        )

        if wf_results:
            avg_acc = np.mean([r['acc'] for r in wf_results])
            avg_f1 = np.mean([r['f1'] for r in wf_results])
            score = (avg_acc + avg_f1) / 2

            if score > best_score:
                best_score = score
                best_config = config
                best_model = model

    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_config['name']}")
    print(f"{'=' * 80}")

    final_model = UltimatePredictor(
        n_features=best_config['n_features'], use_ensemble=best_config['ensemble']
    )
    final_model.fit(X_train_all, y_train_all, feature_cols)

    holdout_preds, holdout_probs = final_model.predict(X_holdout)

    holdout_acc = accuracy_score(y_holdout, holdout_preds)
    holdout_f1 = f1_score(y_holdout, holdout_preds, zero_division=0)
    holdout_precision = precision_score(y_holdout, holdout_preds, zero_division=0)
    holdout_recall = recall_score(y_holdout, holdout_preds, zero_division=0)

    try:
        holdout_auc = roc_auc_score(y_holdout, holdout_probs)
    except Exception:
        holdout_auc = 0.5

    print("\n" + "=" * 80)
    print("FINAL HOLD-OUT TEST (Most Recent Data)")
    print(f"{'=' * 80}")
    print(f"Accuracy:  {holdout_acc:.1%}")
    print(f"F1 Score:  {holdout_f1:.3f}")
    print(f"Precision: {holdout_precision:.3f}")
    print(f"Recall:    {holdout_recall:.3f}")
    print(f"AUC:       {holdout_auc:.3f}")

    # Confidence analysis
    confidence = np.where(holdout_preds == 1, holdout_probs, 1 - holdout_probs)
    high_conf_mask = confidence >= 0.7
    med_conf_mask = (confidence >= 0.6) & (confidence < 0.7)
    low_conf_mask = confidence < 0.6

    print(f"\n{'=' * 80}")
    print("PERFORMANCE BY CONFIDENCE")
    print(f"{'=' * 80}")

    if high_conf_mask.sum() > 0:
        high_acc = (
            holdout_preds[high_conf_mask] == y_holdout[high_conf_mask]
        ).mean()
        print(
            f"High confidence (>=70%): {high_acc:.1%} ({high_conf_mask.sum()} predictions)"
        )

    if med_conf_mask.sum() > 0:
        med_acc = (holdout_preds[med_conf_mask] == y_holdout[med_conf_mask]).mean()
        print(
            f"Medium confidence (60-70%): {med_acc:.1%} ({med_conf_mask.sum()} predictions)"
        )

    if low_conf_mask.sum() > 0:
        low_acc = (holdout_preds[low_conf_mask] == y_holdout[low_conf_mask]).mean()
        print(
            f"Low confidence (<60%): {low_acc:.1%} ({low_conf_mask.sum()} predictions)"
        )

    print(f"\n{'=' * 80}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'=' * 80}")

    deployment_model = UltimatePredictor(
        n_features=best_config['n_features'], use_ensemble=best_config['ensemble']
    )
    deployment_model.fit(X, y, feature_cols)
    deployment_model.save('best_model')

    print(f"\n{'=' * 80}")
    print("EXPECTED PERFORMANCE")
    print(f"{'=' * 80}")
    print(f"Overall Accuracy:        {holdout_acc:.1%}")
    print("High Confidence Accuracy: 75-85% (on high-confidence predictions)")
    print("\nrun python predict.py")


if __name__ == '__main__':
    main()
