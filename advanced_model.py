import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Load the cleaned dataset"""
    print("🔄 Loading cleaned dataset...")
    
    cleaned_file = r"D:\Samples\P1-Consumer\Data\Consumer_Complaints_train_cleaned.csv"
    
    if not os.path.exists(cleaned_file):
        print(f"❌ Cleaned file not found at {cleaned_file}")
        print("Please run data_cleansing.py first!")
        return None
    
    try:
        # Read in chunks for memory efficiency
        chunk_list = []
        chunk_size = 50000
        
        for chunk in pd.read_csv(cleaned_file, chunksize=chunk_size, low_memory=False):
            chunk_list.append(chunk)
        
        df = pd.concat(chunk_list, ignore_index=True)
        print(f"✅ Loaded {len(df)} records with {df.shape[1]} features")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def advanced_feature_engineering(df):
    """Perform advanced feature engineering on the cleaned dataset"""
    print("🔧 Performing advanced feature engineering...")
    
    # Create a copy for processing
    data = df.copy()
    
    # 1. Handle the target variable
    if 'Consumer disputed?' not in data.columns:
        print("❌ Target column 'Consumer disputed?' not found")
        return None, None
    
    # Remove rows where target is missing
    data = data.dropna(subset=['Consumer disputed?'])
    print(f"📊 Dataset shape after removing missing targets: {data.shape}")
    
    # Encode target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(data['Consumer disputed?'])
    print(f"🎯 Target classes: {le_target.classes_}")
    
    # 2. Create feature matrix
    features = []
    feature_names = []
    
    # Categorical features with good completion rates
    categorical_features = ['Product', 'Issue', 'State', 'Submitted via', 
                          'Company response to consumer', 'Timely response?']
    
    print("🏷️ Processing categorical features...")
    for col in categorical_features:
        if col in data.columns:
            # Fill missing values
            data[col] = data[col].fillna('Unknown')
            
            # Label encode
            le = LabelEncoder()
            encoded = le.fit_transform(data[col].astype(str))
            features.append(encoded.reshape(-1, 1))
            feature_names.append(col)
            print(f"   ✅ {col}: {len(le.classes_)} unique values")
    
    # 3. Geographic features
    print("🗺️ Processing geographic features...")
    if 'State' in data.columns:
        # State-based features (already included above)
        pass
    
    if 'ZIP code' in data.columns:
        # Extract ZIP patterns
        data['ZIP_first_digit'] = data['ZIP code'].astype(str).str[0]
        data['ZIP_first_digit'] = data['ZIP_first_digit'].fillna('0')
        
        le_zip = LabelEncoder()
        zip_encoded = le_zip.fit_transform(data['ZIP_first_digit'])
        features.append(zip_encoded.reshape(-1, 1))
        feature_names.append('ZIP_first_digit')
        print(f"   ✅ ZIP_first_digit: {len(le_zip.classes_)} unique values")
    
    # 4. Company-related features
    print("🏢 Processing company features...")
    if 'Company' in data.columns:
        # Use top companies, group others as 'Other'
        top_companies = data['Company'].value_counts().head(50).index
        data['Company_grouped'] = data['Company'].apply(
            lambda x: x if x in top_companies else 'Other'
        )
        data['Company_grouped'] = data['Company_grouped'].fillna('Unknown')
        
        le_company = LabelEncoder()
        company_encoded = le_company.fit_transform(data['Company_grouped'])
        features.append(company_encoded.reshape(-1, 1))
        feature_names.append('Company_grouped')
        print(f"   ✅ Company_grouped: {len(le_company.classes_)} unique values")
    
    # 5. Text features from complaint narrative
    print("📝 Processing text features...")
    if 'Consumer complaint narrative' in data.columns:
        # Only use non-null narratives
        narrative_data = data['Consumer complaint narrative'].fillna('')
        
        # Create TF-IDF features
        try:
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                lowercase=True,
                min_df=10,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            
            tfidf_features = tfidf.fit_transform(narrative_data)
            
            # Add to features
            features.append(tfidf_features.toarray())
            feature_names.extend([f'narrative_tfidf_{i}' for i in range(tfidf_features.shape[1])])
            print(f"   ✅ Narrative TF-IDF: {tfidf_features.shape[1]} features")
            
        except Exception as e:
            print(f"   ⚠️ Could not process narrative text: {e}")
    
    # 6. Date-based features
    print("📅 Processing date features...")
    date_columns = ['Date received', 'Date sent to company']
    for col in date_columns:
        if col in data.columns:
            try:
                data[col] = pd.to_datetime(data[col], errors='coerce')
                
                # Extract date components
                data[f'{col}_year'] = data[col].dt.year
                data[f'{col}_month'] = data[col].dt.month
                data[f'{col}_dayofweek'] = data[col].dt.dayofweek
                
                # Add numeric features
                for component in ['year', 'month', 'dayofweek']:
                    col_name = f'{col}_{component}'
                    if col_name in data.columns:
                        values = data[col_name].fillna(0).astype(int)
                        features.append(values.reshape(-1, 1))
                        feature_names.append(col_name)
                
                print(f"   ✅ {col}: extracted year, month, day_of_week")
                
            except Exception as e:
                print(f"   ⚠️ Could not process {col}: {e}")
    
    # 7. Combine all features
    if features:
        X = np.hstack(features)
        print(f"✅ Final feature matrix: {X.shape}")
        print(f"📋 Total features: {len(feature_names)}")
        return X, y, feature_names
    else:
        print("❌ No features could be created")
        return None, None, None

def train_advanced_models(X, y, feature_names):
    """Train multiple advanced models"""
    print("\n🤖 Training Advanced Models...")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # Scale features for algorithms that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        try:
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_prob,
                'y_true': y_test
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   ✅ AUC Score: {auc_score:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error training {name}: {e}")
    
    # Feature importance for tree-based models
    if 'Random Forest' in results:
        try:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n📊 Top 15 Most Important Features (Random Forest):")
            print(feature_importance.head(15).to_string(index=False))
            
        except Exception as e:
            print(f"Could not display feature importance: {e}")
    
    return results

def create_advanced_visualizations(results):
    """Create comprehensive visualizations"""
    print("\n📊 Creating Advanced Visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Advanced Model Performance Analysis', fontsize=16)
        
        # 1. Accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        auc_scores = [results[name]['auc_score'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, auc_scores, width, label='AUC Score', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels
        for i, (acc, auc) in enumerate(zip(accuracies, auc_scores)):
            axes[0, 0].text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center')
            axes[0, 0].text(i + width/2, auc + 0.01, f'{auc:.3f}', ha='center')
        
        # 2. Best model confusion matrix
        best_model_name = max(results, key=lambda x: results[x]['auc_score'])
        y_true = results[best_model_name]['y_true']
        y_pred = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. ROC Curves
        from sklearn.metrics import roc_curve
        
        for name in model_names:
            if name in results:
                y_true = results[name]['y_true']
                y_prob = results[name]['probabilities']
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = results[name]['auc_score']
                axes[0, 2].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        axes[1, 0].pie(counts, labels=['Not Disputed', 'Disputed'], autopct='%1.1f%%')
        axes[1, 0].set_title('Target Class Distribution')
        
        # 5. Prediction probability distribution
        y_prob = results[best_model_name]['probabilities']
        axes[1, 1].hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Not Disputed', density=True)
        axes[1, 1].hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Disputed', density=True)
        axes[1, 1].set_xlabel('Prediction Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title(f'Probability Distribution - {best_model_name}')
        axes[1, 1].legend()
        
        # 6. Model comparison metrics
        metrics_data = []
        for name in model_names:
            if name in results:
                metrics_data.append([name, results[name]['accuracy'], results[name]['auc_score']])
        
        metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'Accuracy', 'AUC'])
        metrics_df.plot(x='Model', y=['Accuracy', 'AUC'], kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Detailed Metrics Comparison')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('advanced_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Advanced visualizations saved as 'advanced_model_performance.png'")
        
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")

def main():
    """Main execution function for advanced modeling"""
    print("🚀 Advanced Consumer Dispute Prediction Model")
    print("="*60)
    
    # Load cleaned data
    df = load_cleaned_data()
    if df is None:
        return
    
    # Feature engineering
    X, y, feature_names = advanced_feature_engineering(df)
    if X is None:
        return
    
    # Train models
    results = train_advanced_models(X, y, feature_names)
    if not results:
        return
    
    # Create visualizations
    create_advanced_visualizations(results)
    
    # Final summary
    print("\n🎉 FINAL RESULTS")
    print("="*50)
    
    for name, result in results.items():
        print(f"🤖 {name}:")
        print(f"   📊 Accuracy: {result['accuracy']:.4f}")
        print(f"   📊 AUC Score: {result['auc_score']:.4f}")
    
    best_model_name = max(results, key=lambda x: results[x]['auc_score'])
    best_auc = results[best_model_name]['auc_score']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"🎯 Best AUC Score: {best_auc:.4f}")
    print(f"🎯 Best Accuracy: {best_accuracy:.4f}")
    
    print(f"\n✅ Advanced modeling completed successfully!")
    print(f"📊 Trained on {X.shape[0]} samples with {X.shape[1]} features")

if __name__ == "__main__":
    main()
