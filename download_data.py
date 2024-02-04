import urllib
import zipfile


dataset_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
text_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'

extract_dir = '/dataset'


def download_unzip_url(url, extract_dir):
  zip_path, _ = urllib.request.urlretrieve(url)
  with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(extract_dir)

if __name__ == '__main__':
  print('Downloading dataset - Flicker8k')
  download_unzip_url(dataset_url)
  print('Downloading text - Flicker8k')
  download_unzip_url(text_url)
