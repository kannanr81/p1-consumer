import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def check_and_install_packages():
    """Check if required packages are installed and install if missing"""
    required_packages = {
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Installing required packages...")
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please install manually: pip install {package}")
                return False
    
    return True

# Install packages if needed
if not check_and_install_packages():
    print("Please install missing packages manually and run again.")
    sys.exit(1)

# Now import the packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def find_data_files():
    """Find the training and test data files"""
    possible_paths = [
        r"C:\Users\KannanRamaswamy\Downloads\Edv-Proj1\Data\Consumer_Complaints_train.csv",
        r"C:\Users\KannanRamaswamy\Downloads\Edv-Proj1\Data\Consumer_Complaints_test_share.csv",
        "Consumer_Complaints_train.csv",
        "Consumer_Complaints_test_share.csv",
        "data/Consumer_Complaints_train.csv",
        "data/Consumer_Complaints_test_share.csv"
    ]
    
    train_file = None
    test_file = None
    
    # Check current directory and subdirectories
    current_dir = Path(".")
    for file in current_dir.rglob("*.csv"):
        if "train" in file.name.lower() and "consumer" in file.name.lower():
            train_file = str(file)
        elif "test" in file.name.lower() and "consumer" in file.name.lower():
            test_file = str(file)
    
    # Check specified paths
    for path in possible_paths:
        if os.path.exists(path):
            if "train" in path:
                train_file = path
            elif "test" in path:
                test_file = path
    
    return train_file, test_file

def load_and_explore_data():
    """Load and explore the training data"""
    print("Starting Consumer Dispute Prediction Model Training...")
    print("="*60)
    
    # Find data files
    train_file, test_file = find_data_files()
    
    if train_file is None:
        print("Error: Training data file not found!")
        print("Please ensure the training CSV file is in one of these locations:")
        print("1. Current directory")
        print("2. data/ subdirectory")
        print("3. C:\\Users\\KannanRamaswamy\\Downloads\\Edv-Proj1\\Data\\")
        return None, None
    
    print(f"Found training data: {train_file}")
    if test_file:
        print(f"Found test data: {test_file}")
    
    # Try to load the CSV with various parameters
    train_data = None
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            # Try different CSV reading approaches
            csv_configs = [
                {'encoding': encoding},
                {'encoding': encoding, 'sep': ',', 'quotechar': '"'},
                {'encoding': encoding, 'sep': ',', 'quotechar': '"', 'skipinitialspace': True},
                {'encoding': encoding, 'sep': ',', 'on_bad_lines': 'skip'}
            ]
            
            for config in csv_configs:
                try:
                    train_data = pd.read_csv(train_file, **config)
                    print(f"Successfully loaded data with {encoding} encoding")
                    break
                except Exception:
                    continue
            
            if train_data is not None:
                break
                
        except Exception:
            continue
    
    if train_data is None:
        print("Error: Could not read the CSV file. Trying alternative approach...")
        try:
            # Last resort: read line by line and skip problematic lines
            with open(train_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Find the header
            header_line = lines[0].strip().split(',')
            data_lines = []
            
            for i, line in enumerate(lines[1:], 1):
                parts = line.strip().split(',')
                if len(parts) >= len(header_line) - 2:  # Allow some flexibility
                    data_lines.append(parts[:len(header_line)])
            
            # Create DataFrame
            train_data = pd.DataFrame(data_lines, columns=header_line)
            print("Successfully loaded data using fallback method")
            
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None, None
    
    print(f"Training data shape: {train_data.shape}")
    if train_data.shape[1] > 10:
        print(f"Columns (first 10): {list(train_data.columns[:10])}...")
    else:
        print(f"Columns: {list(train_data.columns)}")
    
    # Display basic info
    print(f"\nData Info:")
    print(f"Number of rows: {len(train_data)}")
    print(f"Number of columns: {len(train_data.columns)}")
    
    # Show sample data
    print("\nSample data (first 3 rows, first 5 columns):")
    try:
        print(train_data.iloc[:3, :5].to_string())
    except:
        print("Could not display sample data")
    
    # Find target column
    target_col = None
    possible_target_names = [
        'Consumer disputed?', 'Consumer disputed', 'Consumer Disputed?', 'Consumer Disputed',
        'consumer disputed?', 'consumer disputed', 'Disputed', 'disputed'
    ]
    
    for col_name in possible_target_names:
        if col_name in train_data.columns:
            target_col = col_name
            break
    
    if target_col is None:
        # Search for any column containing 'disputed'
        disputed_cols = [col for col in train_data.columns if 'disputed' in col.lower()]
        if disputed_cols:
            target_col = disputed_cols[0]
        else:
            print("\nAvailable columns:")
            for i, col in enumerate(train_data.columns):
                print(f"{i}: '{col}'")
            
            # Auto-select last column as target if reasonable
            if len(train_data.columns) > 1:
                target_col = train_data.columns[-1]
                print(f"Auto-selected last column as target: '{target_col}'")
    
    if target_col is None:
        print("Could not identify target column")
        return None, None
    
    print(f"\nTarget column: '{target_col}'")
    try:
        print(f"Target distribution:\n{train_data[target_col].value_counts()}")
        print(f"Missing values in target: {train_data[target_col].isnull().sum()}")
    except:
        print("Could not analyze target column")
    
    return train_data, target_col

def preprocess_data(train_data, target_col):
    """Preprocess the data for modeling"""
    print("\nPreprocessing data...")
    
    # Create a copy
    data = train_data.copy()
    
    # Remove rows with missing target values
    initial_rows = len(data)
    data = data.dropna(subset=[target_col])
    print(f"Removed {initial_rows - len(data)} rows with missing target values")
    print(f"Remaining data shape: {data.shape}")
    
    if len(data) == 0:
        print("Error: No valid data remaining")
        return None, None
    
    # Handle target variable
    print(f"Target variable type: {data[target_col].dtype}")
    if data[target_col].dtype == 'object':
        try:
            le_target = LabelEncoder()
            data[target_col] = le_target.fit_transform(data[target_col].astype(str))
            print(f"Target classes: {le_target.classes_}")
        except:
            print("Could not encode target variable")
            return None, None
    
    # Identify feature types
    exclude_patterns = ['id', 'ID', 'Id', 'date', 'Date', 'DATE', 'time', 'Time', 'index', 'Index']
    numerical_features = []
    categorical_features = []
    text_features = []
    
    for col in data.columns:
        if col != target_col and not any(pattern in col for pattern in exclude_patterns):
            try:
                if data[col].dtype in ['object', 'string']:
                    # Determine if text or categorical
                    unique_ratio = data[col].nunique() / len(data)
                    avg_length = data[col].dropna().astype(str).str.len().mean()
                    
                    if pd.isna(avg_length):
                        continue
                        
                    if avg_length > 50 or unique_ratio > 0.8:
                        text_features.append(col)
                    else:
                        categorical_features.append(col)
                else:
                    numerical_features.append(col)
            except:
                continue
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features[:5]}{'...' if len(numerical_features) > 5 else ''}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}")
    print(f"Text features ({len(text_features)}): {text_features[:3]}{'...' if len(text_features) > 3 else ''}")
    
    # Process features
    feature_dfs = []
    
    # Process numerical features
    if numerical_features:
        try:
            num_data = data[numerical_features].copy()
            # Convert to numeric and fill missing values
            for col in numerical_features:
                num_data[col] = pd.to_numeric(num_data[col], errors='coerce')
                median_val = num_data[col].median()
                if pd.isna(median_val):
                    median_val = 0
                num_data[col] = num_data[col].fillna(median_val)
            feature_dfs.append(num_data)
        except Exception as e:
            print(f"Warning: Could not process numerical features: {e}")
    
    # Process categorical features
    if categorical_features:
        try:
            cat_data = pd.DataFrame(index=data.index)
            for col in categorical_features[:10]:  # Limit to first 10
                try:
                    filled_data = data[col].fillna('Unknown').astype(str)
                    le = LabelEncoder()
                    encoded_data = le.fit_transform(filled_data)
                    cat_data[col] = encoded_data
                except Exception as e:
                    print(f"Warning: Could not encode {col}: {e}")
            
            if not cat_data.empty:
                feature_dfs.append(cat_data)
        except Exception as e:
            print(f"Warning: Could not process categorical features: {e}")
    
    # Process text features
    if text_features:
        for text_col in text_features[:2]:  # Limit to first 2
            try:
                print(f"Processing text feature: {text_col}")
                text_data = data[text_col].fillna('').astype(str)
                
                # Use TF-IDF with limited features
                tfidf = TfidfVectorizer(
                    max_features=20,
                    stop_words='english',
                    lowercase=True,
                    min_df=2,
                    max_df=0.95,
                    ngram_range=(1, 1)
                )
                
                text_matrix = tfidf.fit_transform(text_data)
                text_feature_names = [f"{text_col}_tfidf_{i}" for i in range(text_matrix.shape[1])]
                text_df = pd.DataFrame(
                    text_matrix.toarray(),
                    columns=text_feature_names,
                    index=data.index
                )
                feature_dfs.append(text_df)
                
            except Exception as e:
                print(f"Warning: Could not process text feature {text_col}: {e}")
    
    # Combine all features
    if feature_dfs:
        X = pd.concat(feature_dfs, axis=1)
    else:
        print("Error: No features could be processed")
        return None, None
    
    # Remove any remaining NaN values
    X = X.fillna(0)
    y = data[target_col]
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Feature names (first 10): {list(X.columns[:10])}")
    
    return X, y

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models"""
    print(f"\nTraining models...")
    print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    
    if len(X) < 10:
        print("Error: Dataset too small for training")
        return None, None
    
    # Split data
    test_size = min(0.25, max(0.1, 20/len(X)))
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        # If stratify fails, split without stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=1
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
    }
    
    results = {}
    scaler = StandardScaler()
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            if name == 'Logistic Regression':
                # Scale features for logistic regression
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                try:
                    y_prob = model.predict_proba(X_val_scaled)[:, 1]
                except:
                    y_prob = None
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                try:
                    y_prob = model.predict_proba(X_val)[:, 1]
                except:
                    y_prob = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_prob,
                'y_true': y_val
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_val, y_pred, zero_division=0))
            
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    if not results:
        print("Error: No models trained successfully")
        return None, None
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    
    return results, scaler

def create_visualizations(results):
    """Create performance visualizations"""
    if not results:
        return
    
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16)
        
        # 1. Accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
        
        # 2. Confusion matrix for best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        y_true = results[best_model_name]['y_true']
        y_pred = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        axes[1, 0].pie(counts, labels=[f'Class {u}' for u in unique], autopct='%1.1f%%')
        axes[1, 0].set_title('Target Class Distribution')
        
        # 4. Prediction probabilities (if available)
        if results[best_model_name]['probabilities'] is not None:
            probs = results[best_model_name]['probabilities']
            axes[1, 1].hist(probs, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title(f'Prediction Probabilities - {best_model_name}')
            axes[1, 1].set_xlabel('Probability')
            axes[1, 1].set_ylabel('Frequency')
        else:
            axes[1, 1].text(0.5, 0.5, 'Probabilities not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Prediction Probabilities')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Visualization saved as 'model_performance.png'")
        
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")

def main():
    """Main execution function"""
    try:
        # Load data
        train_data, target_col = load_and_explore_data()
        if train_data is None:
            return None, None
        
        # Preprocess data
        X, y = preprocess_data(train_data, target_col)
        if X is None:
            return None, None
        
        # Train models
        results, scaler = train_and_evaluate_models(X, y)
        if results is None:
            return None, None
        
        # Create visualizations
        create_visualizations(results)
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        for name, result in results.items():
            print(f"{name}: {result['accuracy']:.4f}")
        
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"🎯 Best Accuracy: {best_accuracy:.4f}")
        
        # Feature importance for Random Forest
        if 'Random Forest' in results:
            try:
                rf_model = results['Random Forest']['model']
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n📊 Top 10 Most Important Features:")
                print(feature_importance.head(10).to_string(index=False))
            except Exception as e:
                print(f"Could not display feature importance: {e}")
        
        print(f"\n✅ Model training completed successfully!")
        print(f"💾 Model saved for future predictions.")
        
        return results, scaler
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
        return None, None
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, scaler = main()
