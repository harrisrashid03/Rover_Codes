import shutil
import os

def copy_data_file(src_path, dest_path):
    """Copy the data file from source to destination."""
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path))
    shutil.copy(src_path, dest_path)
    print(f"Data copied from {src_path} to {dest_path}")

if __name__ == "__main__":
    # Example usage
    source_path = r'C:\Users\92305\Downloads\Bank Customer Dataset for Churn prediction.xlsx'
    destination_path = r'../data/Bank Customer Dataset for Churn prediction.xlsx'
    copy_data_file(source_path, destination_path)
