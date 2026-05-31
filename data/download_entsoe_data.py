import os
import requests
import sys

BASE_URL = "https://data.open-power-system-data.org/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def download_file(relative_path):
    """
    Downloads a file from the OPSD platform if it doesn't already exist in the local cache.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    local_filename = os.path.basename(relative_path)
    local_path = os.path.join(DATA_DIR, local_filename)

    if os.path.exists(local_path):
        print(f"File {local_filename} already exists at {local_path}. Skipping download.")
        return local_path

    url = BASE_URL + relative_path
    print(f"Downloading from {url} to {local_path}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        written_size = 0
        with open(local_path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
                written_size += len(data)
                if total_size > 0:
                    percent = (written_size / total_size) * 100
                    sys.stdout.write(f"\rProgress: {percent:.2f}% ({written_size/(1024*1024):.2f} MB / {total_size/(1024*1024):.2f} MB)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\rDownloaded {written_size/(1024*1024):.2f} MB")
                    sys.stdout.flush()
        print("\nDownload complete.")
        return local_path
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        print(f"\nError downloading {url}: {e}")
        raise e

if __name__ == "__main__":
    # Test downloader with a smaller file first, or just list files
    # The files available:
    # - time_series/2020-10-06/time_series_60min_singleindex.csv (Hourly)
    # - time_series/2020-10-06/time_series_15min_singleindex.csv (15-min)
    print("OPSD Data Downloader Module initialized.")
    # For testing, we can download the 60min singleindex which is smaller
    # and 15min singleindex for higher frequency.
