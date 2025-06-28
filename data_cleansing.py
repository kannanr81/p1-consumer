import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def data_cleansing_report():
    """Generate a comprehensive data cleansing report and cleaned dataset"""
    
    input_file = r"D:\Samples\P1-Consumer\Data\Consumer_Complaints_train.csv"
    output_file = r"D:\Samples\P1-Consumer\Data\Consumer_Complaints_train_cleaned.csv"
    report_file = r"D:\Samples\P1-Consumer\Data\data_cleansing_report.txt"
    
    print("🧹 Starting Data Cleansing Process...")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found at {input_file}")
        return
    
    try:
        # Read data in chunks to handle large files
        print("📂 Reading data...")
        chunk_list = []
        chunk_size = 10000
        
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-8', low_memory=False):
            chunk_list.append(chunk)
            
        df = pd.concat(chunk_list, ignore_index=True)
        print(f"✅ Successfully loaded {len(df)} rows")
        
    except UnicodeDecodeError:
        try:
            print("🔄 Trying alternative encoding...")
            chunk_list = []
            for chunk in pd.read_csv(input_file, chunksize=chunk_size, encoding='latin-1', low_memory=False):
                chunk_list.append(chunk)
            df = pd.concat(chunk_list, ignore_index=True)
            print(f"✅ Successfully loaded {len(df)} rows with latin-1 encoding")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
    
    # Initialize cleansing report
    report = []
    report.append("DATA CLEANSING REPORT")
    report.append("="*50)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Input file: {input_file}")
    report.append(f"Original dataset shape: {df.shape}")
    report.append("")
    
    print(f"📊 Original dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # 1. BASIC DATA INSPECTION
    print("\n🔍 STEP 1: Basic Data Inspection")
    print("-" * 40)
    
    report.append("1. BASIC DATA INSPECTION")
    report.append("-" * 25)
    report.append(f"Columns: {list(df.columns)}")
    report.append(f"Data types:")
    for col in df.columns:
        report.append(f"  {col}: {df[col].dtype}")
    report.append("")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"💾 Memory usage: {memory_usage:.2f} MB")
    report.append(f"Memory usage: {memory_usage:.2f} MB")
    report.append("")
    
    # 2. MISSING VALUES ANALYSIS
    print("\n🕳️ STEP 2: Missing Values Analysis")
    print("-" * 40)
    
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    report.append("2. MISSING VALUES ANALYSIS")
    report.append("-" * 25)
    
    for col in df.columns:
        missing_count = missing_data[col]
        missing_pct = missing_percentage[col]
        if missing_count > 0:
            print(f"📉 {col}: {missing_count} ({missing_pct:.2f}%) missing values")
            report.append(f"  {col}: {missing_count} ({missing_pct:.2f}%) missing")
    
    if missing_data.sum() == 0:
        print("✅ No missing values found!")
        report.append("  No missing values found!")
    
    report.append("")
    
    # 3. DUPLICATE RECORDS
    print("\n🔄 STEP 3: Duplicate Records Analysis")
    print("-" * 40)
    
    duplicates = df.duplicated().sum()
    print(f"🔍 Found {duplicates} duplicate records")
    report.append("3. DUPLICATE RECORDS")
    report.append("-" * 20)
    report.append(f"Duplicate records found: {duplicates}")
    
    if duplicates > 0:
        print("🧹 Removing duplicate records...")
        df_cleaned = df.drop_duplicates()
        print(f"✅ Removed {duplicates} duplicates. New shape: {df_cleaned.shape}")
        report.append(f"Action: Removed {duplicates} duplicate records")
    else:
        df_cleaned = df.copy()
        report.append("Action: No duplicates to remove")
    
    report.append("")
    
    # 4. DATA TYPE OPTIMIZATION
    print("\n🎯 STEP 4: Data Type Optimization")
    print("-" * 40)
    
    report.append("4. DATA TYPE OPTIMIZATION")
    report.append("-" * 25)
    
    # Optimize numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_cleaned[col].dtype == 'int64':
            if df_cleaned[col].min() >= 0 and df_cleaned[col].max() <= 255:
                df_cleaned[col] = df_cleaned[col].astype('uint8')
                print(f"🔧 Optimized {col}: int64 → uint8")
                report.append(f"  {col}: int64 → uint8")
            elif df_cleaned[col].min() >= -32768 and df_cleaned[col].max() <= 32767:
                df_cleaned[col] = df_cleaned[col].astype('int16')
                print(f"🔧 Optimized {col}: int64 → int16")
                report.append(f"  {col}: int64 → int16")
            elif df_cleaned[col].min() >= -2147483648 and df_cleaned[col].max() <= 2147483647:
                df_cleaned[col] = df_cleaned[col].astype('int32')
                print(f"🔧 Optimized {col}: int64 → int32")
                report.append(f"  {col}: int64 → int32")
    
    # Convert object columns to category where appropriate
    object_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in object_cols:
        unique_ratio = df_cleaned[col].nunique() / len(df_cleaned)
        if unique_ratio < 0.5 and df_cleaned[col].nunique() < 100:  # Categorical threshold
            df_cleaned[col] = df_cleaned[col].astype('category')
            print(f"🔧 Converted {col}: object → category")
            report.append(f"  {col}: object → category")
    
    report.append("")
    
    # 5. TEXT DATA CLEANING
    print("\n📝 STEP 5: Text Data Cleaning")
    print("-" * 40)
    
    report.append("5. TEXT DATA CLEANING")
    report.append("-" * 20)
    
    text_columns = df_cleaned.select_dtypes(include=['object', 'category']).columns
    
    for col in text_columns:
        if df_cleaned[col].dtype == 'object':  # Only clean object columns, not categories
            original_values = df_cleaned[col].copy()
            
            # Remove leading/trailing whitespace
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            
            # Remove extra whitespace
            df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove special characters that might cause issues
            df_cleaned[col] = df_cleaned[col].str.replace(r'[^\w\s\-\.\,\!\?\:\;\(\)]', '', regex=True)
            
            # Check for changes
            changes = (original_values != df_cleaned[col]).sum()
            if changes > 0:
                print(f"🧹 Cleaned {col}: {changes} values modified")
                report.append(f"  {col}: {changes} values cleaned")
    
    report.append("")
    
    # 6. OUTLIER DETECTION
    print("\n📊 STEP 6: Outlier Detection")
    print("-" * 40)
    
    report.append("6. OUTLIER DETECTION")
    report.append("-" * 20)
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            outlier_percentage = (outlier_count / len(df_cleaned)) * 100
            print(f"⚠️ {col}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
            outlier_summary[col] = outlier_count
            report.append(f"  {col}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
    
    if not outlier_summary:
        print("✅ No significant outliers detected in numeric columns")
        report.append("  No significant outliers detected")
    
    report.append("")
    
    # 7. DATA VALIDATION
    print("\n✅ STEP 7: Data Validation")
    print("-" * 40)
    
    report.append("7. DATA VALIDATION")
    report.append("-" * 18)
    
    # Check for empty strings
    empty_strings = 0
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        empty_count = (df_cleaned[col] == '').sum()
        if empty_count > 0:
            empty_strings += empty_count
            print(f"⚠️ {col}: {empty_count} empty strings")
            report.append(f"  {col}: {empty_count} empty strings")
    
    if empty_strings == 0:
        print("✅ No empty strings found")
        report.append("  No empty strings found")
    
    # Check data consistency
    print("✅ Data validation completed")
    report.append("")
    
    # 8. FINAL SUMMARY
    print("\n📋 STEP 8: Final Summary")
    print("-" * 40)
    
    memory_after = df_cleaned.memory_usage(deep=True).sum() / 1024**2
    memory_saved = memory_usage - memory_after
    memory_reduction = (memory_saved / memory_usage) * 100
    
    print(f"📊 Original shape: {df.shape}")
    print(f"📊 Cleaned shape: {df_cleaned.shape}")
    print(f"💾 Memory before: {memory_usage:.2f} MB")
    print(f"💾 Memory after: {memory_after:.2f} MB")
    print(f"💾 Memory saved: {memory_saved:.2f} MB ({memory_reduction:.1f}%)")
    
    report.append("8. FINAL SUMMARY")
    report.append("-" * 15)
    report.append(f"Original shape: {df.shape}")
    report.append(f"Cleaned shape: {df_cleaned.shape}")
    report.append(f"Rows removed: {len(df) - len(df_cleaned)}")
    report.append(f"Memory before: {memory_usage:.2f} MB")
    report.append(f"Memory after: {memory_after:.2f} MB")
    report.append(f"Memory saved: {memory_saved:.2f} MB ({memory_reduction:.1f}%)")
    report.append("")
    
    # 9. SAVE CLEANED DATA
    print("\n💾 STEP 9: Saving Cleaned Data")
    print("-" * 40)
    
    try:
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Cleaned data saved to: {output_file}")
        report.append("9. OUTPUT FILES")
        report.append("-" * 12)
        report.append(f"Cleaned data: {output_file}")
        
        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"📋 Cleansing report saved to: {report_file}")
        report.append(f"Cleansing report: {report_file}")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return
    
    # 10. DATA QUALITY METRICS
    print("\n📈 STEP 10: Data Quality Metrics")
    print("-" * 40)
    
    completeness = ((df_cleaned.count().sum()) / (df_cleaned.shape[0] * df_cleaned.shape[1])) * 100
    consistency = 100 - (empty_strings / (df_cleaned.shape[0] * df_cleaned.shape[1])) * 100
    
    print(f"📊 Data Completeness: {completeness:.2f}%")
    print(f"📊 Data Consistency: {consistency:.2f}%")
    print(f"📊 Duplicate Rate: {(duplicates / len(df)) * 100:.2f}%")
    
    report.append("")
    report.append("10. DATA QUALITY METRICS")
    report.append("-" * 22)
    report.append(f"Data Completeness: {completeness:.2f}%")
    report.append(f"Data Consistency: {consistency:.2f}%")
    report.append(f"Duplicate Rate: {(duplicates / len(df)) * 100:.2f}%")
    
    print("\n🎉 Data Cleansing Completed Successfully!")
    print("="*60)
    
    return df_cleaned, '\n'.join(report)

def generate_data_profile(df):
    """Generate a detailed data profile"""
    print("\n📊 Generating Data Profile...")
    
    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values': df.nunique().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Statistical summary for numeric columns
    numeric_summary = df.describe().to_dict()
    
    # Sample values for each column
    sample_values = {}
    for col in df.columns:
        sample_values[col] = df[col].dropna().head(5).tolist()
    
    profile['numeric_summary'] = numeric_summary
    profile['sample_values'] = sample_values
    
    return profile

if __name__ == "__main__":
    # Run data cleansing
    cleaned_data, report = data_cleansing_report()
    
    if cleaned_data is not None:
        # Generate additional profile
        profile = generate_data_profile(cleaned_data)
        
        print(f"\n📋 Data Profile Summary:")
        print(f"   Shape: {profile['shape']}")
        print(f"   Memory: {profile['memory_usage_mb']:.2f} MB")
        print(f"   Columns: {len(profile['columns'])}")
