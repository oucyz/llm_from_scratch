# import urllib.request

from .gpt_download import download_and_load_gpt2

if __name__ == "__main__":
    # url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"
    # filename = url.split("/")[-1]
    # urllib.request.urlretrieve(url, filename)

    settings, params = download_and_load_gpt2("355M", "./output/gpt2")
    print("settings:", settings)
    print("params:", params.keys())
