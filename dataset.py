import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/"
FOLDERS = ["training_setA/", "training_setB/"]

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_folder(folder):
    url = urljoin(BASE_URL, folder)
    print("Scanning:", url)

    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    links = soup.find_all("a")

    for link in links:
        file_name = link.get("href")

        if file_name.endswith(".psv"):
            file_url = urljoin(url, file_name)
            save_path = os.path.join(SAVE_DIR, file_name) 

            if os.path.exists(save_path):
                print("Skipping:", file_name)
                continue

            print("Downloading:", file_name)

            response = requests.get(file_url, stream=True)

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)

for folder in FOLDERS:
    download_folder(folder)

print("Download Complete")
