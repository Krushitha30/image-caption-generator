import pandas as pd
import re
from collections import defaultdict

# Load captions
df = pd.read_csv('captions.txt')

def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)  # remove punctuation
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption

def add_tokens(caption):
    return '<start> ' + caption + ' <end>'

df['cleaned_caption'] = df['caption'].apply(clean_caption)
df['tokenized_caption'] = df['cleaned_caption'].apply(add_tokens)

print(df[['image', 'caption', 'tokenized_caption']].head())
img_to_captions = defaultdict(list)
for _, row in df.iterrows():
    img_to_captions[row['image']].append(row['tokenized_caption'])

# Sanity check
sample_img = list(img_to_captions.keys())[0]
print(f"\nImage: {sample_img}")
print(f"Captions: {img_to_captions[sample_img]}")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 5000  # Flickr8k has a smaller vocab than COCO, 5000 is a good cap

all_captions = df['tokenized_caption'].tolist()

tokenizer = Tokenizer(num_words=VOCAB_SIZE,
                       oov_token="<unk>",
                       filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

tokenizer.fit_on_texts(all_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

print(f"\nVocabulary size: {len(tokenizer.word_index)}")
sequences = tokenizer.texts_to_sequences(all_captions)

max_length = max(len(seq) for seq in sequences)
print(f"Max caption length: {max_length}")

cap_vector = pad_sequences(sequences, maxlen=max_length, padding='post')

print(f"\nShape of caption vectors: {cap_vector.shape}")
print(f"Example padded sequence: {cap_vector[0]}")
import pickle

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\nTokenizer saved as tokenizer.pkl")
