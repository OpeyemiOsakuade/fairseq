import pandas as pd

# Load the CSV data
file_path = '/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/all_vectors/syllables_train_dev_yor_vowels_with_tones_correct.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, header=None)

# Define the consonants and ending patterns
consonants = {'b', 'd', 'f', 'g', 'ɡb','j', 'dz', 'dʒ', 'k', 'kp', 'nl', 'l', 'm', 'n', 'r', 's','ʒ','sh', 't', 'v','w','y'}
ending_patterns = ['LL', 'HH']

# Function to process each word
def process_word(word):
    # Check for consonants at the beginning
    for consonant in consonants:
        if word.startswith(consonant):
            word = word[len(consonant):]
            break
    
    # Check for ending patterns
    for pattern in ending_patterns:
        if word.endswith(pattern):
            word = word[:-len(pattern)] + pattern[0]
            break
    
    return word

# Apply the function to each word in the dataframe
data[0] = data[0].apply(process_word)

# Save the cleaned data back to CSV
output_file_path = '/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/all_vectors/syllables_train_dev_yor_vowels_with_tones_correct_1.csv'  # Replace with your desired output file path
data.to_csv(output_file_path, index=False, header=False)

print(f"Data has been processed and saved to {output_file_path}")
