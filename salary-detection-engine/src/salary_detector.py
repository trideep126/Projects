import pandas as pd
import numpy as np
from datetime import datetime
import msoffcrypto
import io
import getpass 
import re
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import warnings
import json
import os
warnings.filterwarnings('ignore')


config = {
  'SKIP_ROWS': 17,
  'COL_DATE': 'Date',
  'COL_DETAILS': 'Details',
  'COL_CREDIT': 'Credit',
  'COL_DEBIT': 'Debit',
  'COL_BALANCE': 'Balance'
}

assumptions = {
    'min_occurences': 3,
    'min_months': 3,
    'regularity_threshold': 0.6,
    'amount_variance_threshold': 0.15,
    'min_salary_amt': 10000,
    'max_salary_amt': 400000
}

##Helper Functions
def extract_statement(file_path: str, file_password: str) -> pd.DataFrame:
  if msoffcrypto is None:
    raise ImportError("msoffcrypto is required to decrypt encrypted Excel files. Install with: pip install msoffcrypto-tool")

  path = file_path
  password = file_password

  if not password:
    try:
      password = getpass.getpass(f"Enter password for '{path}': ")
    except Exception:
      password = ''

  decrypted_file = io.BytesIO()

  with open(path, 'rb') as file:
    office_file = msoffcrypto.OfficeFile(file)
    office_file.load_key(password=password)
    office_file.decrypt(decrypted_file)

  decrypted_file.seek(0)

  df = pd.read_excel(
      decrypted_file,
      usecols = 'A:F',
      skiprows=17,
      nrows=130,
      header=0
  )

  return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
  try:
    df = df.copy()

    df = df.rename(columns ={
      config['COL_DATE']: 'date',
      config['COL_DETAILS']: 'details',
      config['COL_CREDIT']: 'credit',
      config['COL_DEBIT']: 'debit',
      config['COL_BALANCE']: 'balance'
    })

    df.columns = df.columns.str.lower()

    if 'ref no/cheque no' in df.columns:
      df.drop(columns=['ref no/cheque no'], inplace=True)
    else:
      warnings.warn("'Ref No/Cheque No' column not found")

    df = df[(df['credit'].notna() | df['debit'].notna())]

    if len(df) == 0:
      raise ValueError("No valid transactions found after filtering")

    required_cols = ['date','details','credit','debit','balance']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
      raise ValueError(f"Required columns not found : {missing_cols}")

    df['date'] = pd.to_datetime(df['date'],dayfirst=True, errors='coerce')

    if df['date'].isna().sum() > 0:
      df = df.dropna(subset=['date'])
      print(f"Dropped {df['date'].isna().sum()} rows, they had invalid dates")

    df['details'] = df['details'].astype(str)

    df['credit'] = pd.to_numeric(df['credit'], errors='coerce')
    df['debit'] = pd.to_numeric(df['debit'],errors = 'coerce')
    df['balance'] = pd.to_numeric(df['balance'],errors='coerce')

    return df

  except Exception as e:
    print(f"Error during data preparation {e}")
    raise
  

def get_transaction_type(details: str) -> str:
  if pd.isna(details):
    return 'UNKNOWN'

  details_upper = str(details).upper()

  if 'NEFT' in details_upper: return 'NEFT'
  if 'RTGS' in details_upper: return 'RTGS'
  if 'IMPS' in details_upper: return 'IMPS'
  if 'UPI' in details_upper: return 'UPI'
  if 'ATM' in details_upper: return 'ATM'
  if 'CASH DEPOSIT' in details_upper: return 'CASH DEPOSIT'
  if any(word in details_upper for word in ['CHQ','CHEQUE']):
    return 'CHEQUE'
  if any(word in details_upper for word in ['SALARY','SAL']):
    return 'SALARY'
  if any(word in details_upper for word in ['CASH','DEP']):
    return 'CASH'

  return 'OTHER'


def find_salary_candidates(credit_df: pd.DataFrame)->Optional[pd.DataFrame]:
  threshold = credit_df['credit'].quantile(0.5)
  large_credits = credit_df[credit_df['credit'] > threshold].copy()

  if len(large_credits) < assumptions['min_occurences']:
    return None

  large_credits = large_credits.sort_values('credit')
  clusters = []
  current_cluster = [large_credits.iloc[0]]

  for i in range(1, len(large_credits)):
    current_amount = large_credits.iloc[1]['credit']
    cluster_mean = np.mean([c['credit'] for c in current_cluster])

    if abs(current_amount - cluster_mean) <= 0.15 * cluster_mean:
      current_cluster.append(large_credits.iloc[i])
    else:
      if len(current_cluster) >= assumptions['min_occurences']:
        clusters.append(current_cluster)
      current_cluster = [large_credits.iloc[i]]

  if len(current_cluster) >= assumptions['min_occurences']:
    clusters.append(current_cluster)

  if not clusters:
    return None

  largest_cluster = max(clusters, key=len)
  return pd.DataFrame(largest_cluster)

def calculate_regularity_score(transaction_df: pd.DataFrame) -> float:
  if len(transaction_df) < 2:
    return 0

  amount_cv = transaction_df['credit'].std() / transaction_df['credit'].mean()
  amount_score = max(0, 1 - amount_cv)

  days = transaction_df['day'].values
  day_std = np.std(days)
  time_score = max(0, 1 - (day_std/15))

  months = transaction_df['month'].nunique()
  total_months = (transaction_df['date'].max() - transaction_df['date'].min()).days / 30
  occurence_score = min(1.0, months/max(1, total_months))

  weighted_avg = 0.4 * amount_score + 0.4 * time_score + 0.2 * occurence_score
  return weighted_avg


def check_source_consistency(clustered_df: pd.DataFrame) -> float:
  if len(clustered_df) <2:
    return 0.0

  descriptions = clustered_df['details'].astype(str).tolist()

  similarity_scores = []
  base_desc = descriptions[0]

  for desc in descriptions[1:]:
    ratio = SequenceMatcher(None, base_desc, desc).ratio()
    similarity_scores.append(ratio)

  return np.mean(similarity_scores)


def get_credit_summary(credit_df: pd.DataFrame) -> Dict:
  return {
      'total_credits': len(credit_df),
      'total_months': credit_df['month'].nunique(),
      'avg_credits_per_month': round(len(credit_df) / credit_df['month'].nunique(), 1),
      'credit_amt_range': (int(credit_df['credit'].min()), int(credit_df['credit'].max())),
      'median_credit': int(credit_df['credit'].median())
  }

def error_handling(text: str, confidence:float= 0.0) -> Dict:
  return {
      'is_salaried': False,
      'estimated_salary': None,
      'reason': text,
      'details': {},
      'confidence_score': confidence
  }

def detect_salary(df: pd.DataFrame) -> Dict:
  if df is None or len(df) == 0:
    return error_handling("No valid transaction found")

  credits = df[df['credit'].notna()].copy()

  if len(credits) == 0:
    return error_handling("No credit transactions found")

  credits['month'] = credits['date'].dt.to_period('M')
  credits['day'] = credits['date'].dt.day
  credits['txntype'] = credits['details'].apply(get_transaction_type)

  n_months = credits['month'].nunique()
  if n_months < assumptions['min_months']:
    return error_handling(f"Insufficient Data: {n_months} months of data")

  salary_candidates = find_salary_candidates(credits)

  if salary_candidates is None or len(salary_candidates) < assumptions['min_occurences']:
    return {
        'is_salaried': False,
        'estimated_salary': None,
        'reason': 'No regular high-value credits detected',
        'details': get_credit_summary(credits)
    }


  regularity_score = calculate_regularity_score(salary_candidates)
  amount_cv = salary_candidates['credit'].std() / salary_candidates['credit'].mean()
  amount_consistency_score = max(0.0, 1.0 - amount_cv)

  source_consistency = check_source_consistency(salary_candidates)

  confidence = round(np.mean([regularity_score,amount_consistency_score,source_consistency]), 2)

  is_salaried = (
      regularity_score >= assumptions['regularity_threshold'] and
      amount_cv <= assumptions['amount_variance_threshold'] and
      source_consistency > 0.8
  )

  if is_salaried:
    estimated_salary = int(salary_candidates['credit'].median())

    return {
        'is_salaried': True,
        'confidence_score': confidence,
        'estimated_salary': estimated_salary,
        'salary_range': (
        int(salary_candidates['credit'].min()),
        int(salary_candidates['credit'].max()))
    }

  else:
    reason_parts = []
    if regularity_score < assumptions["regularity_threshold"]:
      reason_parts.append(f'regularity score too low (score: {regularity_score:.2f}, threshold: {assumptions["regularity_threshold"]})')
    if amount_cv > assumptions["amount_variance_threshold"]:
      reason_parts.append(f'amount variance too high (CV: {amount_cv:.2f}, threshold: {assumptions["amount_variance_threshold"]})')

    reason_str = ''
    if len(reason_parts) == 1:
        reason_str = f'Credits not regular enough: {reason_parts[0]}'
    elif len(reason_parts) == 2:
        reason_str = f'Credits not regular enough: {reason_parts[0]} and {reason_parts[1]}'
    else:
        reason_str = f'Credits not regular enough (score: {regularity_score:.2f}, threshold: {assumptions["regularity_threshold"]})'

    return {
        'is_salaried': False,
        'confidence_score': confidence,
        'estimated_salary': None,
        'reason': reason_str,
        'details': get_credit_summary(credits),
    }


def generate_test_data(user_type: str = 'salaried', months: int = 6) -> pd.DataFrame:

    transactions = []
    start_date = pd.Timestamp('2024-01-01')
    current_balance = 10000

    if user_type == 'salaried':
        base_salary = 40000
        for month in range(months):
            salary_date = start_date + pd.DateOffset(months=month)

            credit_val = base_salary + np.random.normal(0, 500)
            current_balance += credit_val
            transactions.append({
                'Date': salary_date.strftime('%d/%m/%Y'),
                'Details': 'NEFT/SALARY/XYZ_CORP/INR',
                'Credit': credit_val,
                'Debit': np.nan,
                'Balance': current_balance
            })

            for _ in range(np.random.randint(3, 8)):
                noise_date = salary_date + pd.Timedelta(days=np.random.randint(1, 28))

                credit_noise = np.nan
                debit_noise = np.nan

                if np.random.random() > 0.7:
                    credit_noise = np.random.uniform(100, 3000)
                    current_balance += credit_noise

                if np.random.random() > 0.3:
                    debit_noise = np.random.uniform(100, 5000)
                    current_balance -= debit_noise

                transactions.append({
                    'Date': noise_date.strftime('%d/%m/%Y'),
                    'Details': f'UPI/MERCHANT{np.random.randint(1,100)}',
                    'Credit': credit_noise,
                    'Debit': debit_noise,
                    'Balance': current_balance
                })

    elif user_type == 'freelancer':
        for month in range(months):
            base_date = start_date + pd.DateOffset(months=month)

            for _ in range(np.random.randint(2, 6)):
                payment_date = base_date + pd.Timedelta(days=np.random.randint(1, 28))

                credit_val = np.random.uniform(5000, 50000)
                current_balance += credit_val
                transactions.append({
                    'Date': payment_date.strftime('%d/%m/%Y'),
                    'Details': f'UPI/CLIENT{np.random.randint(1,20)}/PAYMENT',
                    'Credit': credit_val,
                    'Debit': np.nan,
                    'Balance': current_balance
                })

    elif user_type == 'student':
        for month in range(months):
            base_date = start_date + pd.DateOffset(months=month)
            for _ in range(np.random.randint(1, 4)):
                txn_date = base_date + pd.Timedelta(days=np.random.randint(1, 28))

                credit_val = np.nan
                debit_val = np.nan

                if np.random.random() > 0.5:
                    credit_val = np.random.uniform(500, 5000)
                    current_balance += credit_val

                if np.random.random() > 0.3:
                    debit_val = np.random.uniform(100, 2000)
                    current_balance -= debit_val

                transactions.append({
                    'Date': txn_date.strftime('%d/%m/%Y'),
                    'Details': 'UPI/PARENT/TRANSFER' if np.random.random() > 0.5 else 'ATM/CASH_DEPOSIT',
                    'Credit': credit_val,
                    'Debit': debit_val,
                    'Balance': current_balance
                })

    transactions.sort(key=lambda x: datetime.strptime(x['Date'], '%d/%m/%Y'))

    return pd.DataFrame(transactions)


if __name__ == "__main__":
    import argparse
    
    config_file = 'config.json'
    default_file = None
    default_password = None
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                default_file = config_data.get('file_path')
                default_password = config_data.get('password')
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
    
    parser = argparse.ArgumentParser(description='Predict Salary from Bank Statement')
    parser.add_argument('--file', type=str, default=default_file, help='Path to the bank statement file (default: from config.json)')
    parser.add_argument('--password', type=str, default=default_password, help='File password (default: from config.json)')
    parser.add_argument('--generate', action='store_true', help='Generate synthetic data instead of reading a file')
    parser.add_argument('--user-type', type=str, choices=['salaried', 'freelancer', 'student'], default='salaried', help='Type of synthetic user to generate (used with --generate)')
    parser.add_argument('--months', type=int, default=6, help='Number of months of synthetic data to generate (used with --generate)')
    parser.add_argument('--save-generated', type=str, default=None, help='If set while using --generate, save the generated data to this CSV path')

    args = parser.parse_args()


    if not args.generate and not args.file:
      parser.error("--file argument is required or must be set in config.json, unless --generate is used")

    try:
      if args.generate:
        print(f"Generating synthetic data: type={args.user_type}, months={args.months}...")
        df = generate_test_data(user_type=args.user_type, months=args.months)
        if args.save_generated:
          try:
            df.to_csv(args.save_generated, index=False)
            print(f"Saved generated data to {args.save_generated}")
          except Exception as e:
            print(f"Warning: could not save generated data: {e}")
      else:
        print(f"Processing {args.file}...")
        if args.file.endswith('.xlsx') and args.password:
          df = extract_statement(args.file, args.password)
        else:
          df = pd.read_excel(args.file) if args.file.endswith('.xlsx') else pd.read_csv(args.file)

      clean_df = prepare_data(df)

      result = detect_salary(clean_df)

      print("\n")
      print("RESULT")
      print(f"Is Salaried: {result['is_salaried']}")
      if result['is_salaried']:
        print(f"Estimated Salary: ₹{result['estimated_salary']}")
        print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
      else:
        print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
        print(f"Reason: {result['reason']}")
      print("\n")

    except Exception as e:
      print(f"Error: {str(e)}")