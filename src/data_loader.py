import pandas as pd

def load_parallel_data(file_path):
    """
    Load parallel text data from a tab-separated file
    Returns a pandas DataFrame with Tibetan and English columns
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    tibetan, english = parts
                    data.append({
                        'tibetan': tibetan.strip(),
                        'english': english.strip()
                    })
    
    return pd.DataFrame(data)

def explore_data(df):
    """Print basic statistics about the dataset"""
    print(f"Dataset contains {len(df)} parallel examples")
    
    # Calculate average lengths
    tibetan_char_lengths = df['tibetan'].str.len()
    english_char_lengths = df['english'].str.len()
    
    print(f"Average Tibetan text length: {tibetan_char_lengths.mean():.2f} characters")
    print(f"Average English text length: {english_char_lengths.mean():.2f} characters")
    
    # Print a few examples
    print("\nSample entries:")
    print(df.head(3))

if __name__ == "__main__":
    # Load the sample data we created earlier
    file_path = 'data/raw/tibetan_english_sample.txt'
    df = load_parallel_data(file_path)
    explore_data(df)