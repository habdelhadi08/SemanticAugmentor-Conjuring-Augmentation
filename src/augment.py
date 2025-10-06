import pandas as pd 
import nlpaug.augmenter.word as naw
import nltk

# Download required resources once
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

#Load csv file
df=pd.read_csv('data/text.csv')
print(df)
# Initialize the synonym augmenter
aug = naw.SynonymAug(aug_src='wordnet')
# Apply augmentation to the 'text' column
df['augmented_text'] = df['text'].apply(lambda x: aug.augment(x))
print(df)
# Save the augmented data to a new CSV file
df.to_csv('data/augmented_text.csv', index=False)
print("Augmented data saved to 'data/augmented_text.csv'")



