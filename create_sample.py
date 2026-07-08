import pandas as pd
import os

df = pd.read_csv('captions.txt')

sample_files = os.listdir('sample_images')
sample_df = df[df['image'].isin(sample_files)]

sample_df.to_csv('sample_captions.txt', index=False)
print(f"Saved {len(sample_df)} sample captions")