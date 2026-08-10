import pymupdf as fitz
import json

print("Reading PDF...")
doc = fitz.open('../ilovepdf_merged.pdf')
chunks = []

for page_num, page in enumerate(doc):
    text = page.get_text().strip()
    if not text:
        continue
    words = text.split()
    for i in range(0, len(words), 150):
        chunk_text = ' '.join(words[i:i+200])
        if chunk_text:
            chunks.append({'text': chunk_text, 'page': page_num + 1})

with open('chunks.json', 'w') as f:
    json.dump(chunks, f)

print(f'Done! {len(chunks)} chunks saved.')