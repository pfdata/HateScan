import pandas as pd
import spacy

def load_model(language):
    """Load the spaCy model based on the language specified.
    
    Args:
        language (str): Language of the transcripts ('english', 'german', 'polish').
        
    Returns:
        spacy.language.Language: Loaded spaCy model for the specified language.
    
    Raises:
        ValueError: If an unsupported language is specified.
    """
    # Mapping of supported languages to their spaCy model names
    model_names = {
        'english': 'en_core_web_trf',
        'german': 'de_dep_news_trf',
        'polish': 'pl_core_news_lg'
    }
    
    if language.lower() not in model_names:
        raise ValueError(f"Unsupported language '{language}'. Supported languages are: {list(model_names.keys())}")
    
    return spacy.load(model_names[language.lower()])

def pos_tag_transcripts(df, transcript_col='transcript', language='german', output_path=None):
    """Replaces transcripts with sequences of part-of-speech tags based on the specified language without modifying the original DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the transcripts.
        transcript_col (str): Column name containing the transcripts. Defaults to 'transcript'.
        language (str): Language of the transcripts. Defaults to 'german'.
        output_path (str, optional): Path to save the modified DataFrame. Defaults to None.
        
    Returns:
        pd.DataFrame: Modified DataFrame with transcripts replaced by POS tag sequences, leaving the original DataFrame unchanged.
    """
    # Create a copy of the DataFrame to ensure the original is not modified
    df_copy = df.copy()
    
    nlp = load_model(language)
    
    def extract_pos_tags(transcript):
        """Extract POS tags from a transcript using the loaded spaCy model."""
        return ' '.join([token.tag_ for token in nlp(transcript)])
    
    # Apply the POS tag extraction on the copy of the DataFrame
    df_copy[transcript_col] = df_copy[transcript_col].apply(extract_pos_tags)
    
    if output_path:
        # Save the modified copy to the specified output path
        df_copy.to_csv(output_path, index=False)
    
    return df_copy


def extract(df, transcript_col='transcript', language='german', output_path=None):
    """Calculates relative occurrences of POS tags and updates the DataFrame with these as columns.
    
    Args:
        df (pd.DataFrame): DataFrame with transcript presented as POS tag sequence.
        transcript_col (str): Column name containing the transcripts with the POS tag sequence. Defaults to 'transcript'.
        language (str): Language of the transcripts. Defaults to 'german'.
        output_path (str, optional): Path to save the modified DataFrame. Defaults to None.
        
    Returns:
        pd.DataFrame: DataFrame with POS tag relative occurrences as columns.
    """
    nlp = load_model(language)
    pos_tags = list(nlp.get_pipe("tagger").labels)
    
    pos_occurrences = {tag: [] for tag in pos_tags}
    
    for transcript in df[transcript_col]:
        doc = nlp(transcript)
        total_tags = len(doc)
        
        tag_counts = {tag: 0 for tag in pos_tags}
        for token in doc:
            if token.text in tag_counts:
                tag_counts[token.text] += 1
        
        for tag in pos_tags:
            pos_occurrences[tag].append(tag_counts[tag] / total_tags if total_tags > 0 else 0)
    
    pos_df = pd.DataFrame(pos_occurrences)
    result_df = pd.concat([df.reset_index(drop=True), pos_df], axis=1).drop(columns=[transcript_col])
    
    if output_path:
        result_df.to_csv(output_path, index=False)
    
    return result_df