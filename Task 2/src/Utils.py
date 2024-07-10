import shutil
import os

def copy_transactions_data(src_path, dest_path):
    """Copy the transactions data file from source to destination."""
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path))
    shutil.copy(src_path, dest_path)
    print(f"Data copied from {src_path} to {dest_path}")

if __name__ == "__main__":
    # Example usage
    source_path = r'C:\Users\92305\Downloads\Transactions Data.csv'
    destination_path = r'../data/credit_card_transactions.csv'
    copy_transactions_data(source_path, destination_path)
