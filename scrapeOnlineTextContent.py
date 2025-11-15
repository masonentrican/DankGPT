import urllib.request

# Variables
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "the-verdict.txt"

# HTTP Request
_ = urllib.request.urlretrieve(url,file_path)

# Write the file
with open(file_path,"r",encoding="utf-8") as f:
    raw_text = f.read()

# Debug
print("Total chars:", len(raw_text))
print(raw_text[:99])
