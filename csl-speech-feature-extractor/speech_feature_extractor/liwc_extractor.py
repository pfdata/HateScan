import collections
import nltk
import pandas as pd
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import transcript as transcriber

nltk.download('punkt')

def load_liwc_dictionary(liwc_path):
    """
    Loads the LIWC dictionary from a file and constructs a lookup structure.

    Args:
        liwc_path (str): Path to the LIWC dictionary file.

    Returns:
        dict: A dictionary containing LIWC categories, lookup patterns, and mapping of categories to numbers.
    """
    with open(liwc_path, 'r') as file:
        categories = collections.OrderedDict()
        file.readline()  # Skip the first '%' line
        line = file.readline()
        lookup = []
        while '%' not in line:
            key, value = line.split('\t')
            categories[key] = {'name': value.strip(), 'words': []}
            line = file.readline()
        category_indices = list(categories.keys())
        file.readline()  # Skip the second '%' line

        while line:
            parts = line.strip().split('\t')
            pattern = '^' + parts[0].replace('*', '.*') + '$'
            indices = [category_indices.index(part) for part in parts[1:]]
            lookup.append((re.compile(pattern), indices))
            for cat in parts[1:]:
                categories[cat]['words'].append((parts[0], pattern))
            line = file.readline()

    return {'categories': categories, 'lookup': lookup, 'cat_to_num': category_indices}

class LIWCExtractor:
    """
    A class to extract LIWC features from text documents.
    """
    def __init__(self, liwc_path):
        """
        Initializes the LIWCExtractor with a given LIWC dictionary.

        Args:
            liwc_path (str): Path to the LIWC dictionary file.
        """
        self.tokenizer = nltk.word_tokenize
        self.dictionary = load_liwc_dictionary(liwc_path)
        self.lookup = self.dictionary['lookup']
        self.categories = self.dictionary['categories']
        self.category_indices = self.dictionary['cat_to_num']
        self.non_dict_tokens = []

    def extract_features(self, dataframe, normalize=True, output_path=None):
        """
        Extracts LIWC features from a dataframe containing audio paths and transcripts.

        Args:
            dataframe (pd.DataFrame): A DataFrame with columns ['audio_path', 'transcript'].
            normalize (bool): Whether to normalize the transcript text.
            output_path (str): Optional path to save the extracted features as a CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted features along with the audio paths.
        """
        features_list = []
        # Normalize text if required
        if normalize:
            dataframe = transcriber.normalize_transcripts(dataframe)
        for _, row in dataframe.iterrows():
            text = row['transcript']
            features = self.extract_from_transcript(text)
            features_list.append(features)

        features_df = pd.DataFrame(features_list, columns=self.get_category_names())
        features_df.insert(0, 'audio_path', dataframe['audio_path'])

        if output_path:
            features_df.to_csv(output_path, index=False)

        return features_df

    def extract_from_transcript(self, text):
        """
        Extracts LIWC features from a single transcript.

        Args:
            text (str): The transcript to extract features from.

        Returns:
            list: Normalized LIWC features from transcript.
        """
        tokens = self.tokenizer(text)
        features = [0] * (len(self.categories) + 5)  # Initialize feature counts
        total_words_pos = len(self.categories)
        features[total_words_pos] = len(tokens)  # Total word count

        for token in tokens:
            if len(token) > 6:
                features[len(self.categories) + 1] += 1  # Six letter words
            found_in_dict = False
            for pattern, cat_indices in self.lookup:
                if pattern.findall(token):
                    found_in_dict = True
                    for index in cat_indices:
                        features[index] += 1
            if found_in_dict:
                features[len(self.categories) + 2] += 1  # Words found in LIWC dictionary
            else:
                self.non_dict_tokens.append(token)

        # Normalize features if normalization is enabled
        if features[total_words_pos] > 0:
            features = [x / features[total_words_pos] for x in features]
        return features

    def get_category_names(self):
        """
        Retrieves the names of the LIWC categories.

        Returns:
            list: A list of category names.
        """
        names = [category['name'] for category in self.categories.values()]
        names += ['total_words', 'six_letter_words', 'dict_words', 'punctuation', 'emoticons']
        return names

def extract(transcripts, language='German', output_path=None, normalize=False):
    """
    Extracts LIWC features from transcripts in a specified language.

    Args:
        transcripts (pd.DataFrame): DataFrame containing transcripts and corresponding audio paths.
        language (str): The language of the transcripts, supports 'German' or 'English'.
        output_path (str): Optional path to save the extracted features as a CSV file.
        normalize (bool): Whether to normalize the transcript text.
    """
    if language not in ["German", "English"]:
        raise ValueError("Only German or English languages are supported.")

    extractor = LIWCExtractor(liwc_path=f'{os.path.dirname(os.path.abspath(__file__))}/data/LIWC_{language}.dic')
    return extractor.extract_features(dataframe=transcripts, normalize=normalize, output_path=output_path)
