import os
import pycountry
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

filters = ['parking']
filters_pos = ['park', 'parked']


def check_filters(text):
    """Check if the text contains the filters or the filters_pos"""
    text = text.lower().strip()
    if any(f in text for f in filters):
        return True
    elif any(f in text.split() for f in filters_pos):
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        for word, pos in pos_tags:
            if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ') and (word in filters_pos):
                return True
        return False
    else:
        return False


def filter_data(filename1, filename2):
    """Filter the dataframes based on the filters and filters_pos"""
    # read dataframe 1 and 2
    # process dataframe 2 in chunks
    chunksize = 10000
    filtered_df2_chunks = []
    for chunk in pd.read_json(filename2, compression='gzip', orient='records', lines=True, chunksize=chunksize):
        chunk['text'] = chunk['text'].astype(str)
        chunk = chunk.dropna(subset='text')
        filtered_chunk = chunk[chunk['text'].apply(check_filters)]
        filtered_df2_chunks.append(filtered_chunk)
    filtered_df2 = pd.concat(filtered_df2_chunks)
    print(f"Filter file {filename2} ----- after: {len(filtered_df2)}")
    filtered_df2['gmap_id'] = filtered_df2['gmap_id'].astype(str)
    gmap_list = list(set(filtered_df2['gmap_id'].tolist()))
    # filter dataframe 1
    df1 = pd.read_json(filename1, compression='gzip', orient='records', lines=True)
    df1['gmap_id'] = df1['gmap_id'].astype(str)
    filtered_df1 = df1[df1['gmap_id'].isin(gmap_list)]
    print(f"Filter file {filename1} ----- before: {len(df1)} after: {len(filtered_df1)}")    
    return filtered_df1, filtered_df2
    
    
def write_data(filtered_df1, filename1, filtered_df2, filename2):
    """Write the filtered dataframes to csv files"""
    write_filename1 = filename1.replace('meta', 'filtered_meta').split('.json')[0] + '.csv'
    write_filename1 = os.path.join(save_meta_folder, write_filename1)
    write_filename2 = filename2.replace('review', 'filtered_review').split('.json')[0] + '.csv'
    write_filename2 = os.path.join(save_review_folder, write_filename2)
    filtered_df1.to_csv(write_filename1, index=False)
    print(f"Finish writing file: {write_filename1}")
    filtered_df2.to_csv(write_filename2, index=False)
    print(f"Finish writing file: {write_filename2}\n")


if __name__ == '__main__':
    countries = pycountry.countries
    excluded_states = []  # Add states to exclude
    us_states = [state.name.replace(" ", "_") if " " in state.name else state.name for state in pycountry.subdivisions.get(country_code='US') if state.name not in excluded_states]
    us_states = sorted(us_states)[:25]
    print(us_states)

    read_folder = "data-google-map"
    filenames = os.listdir(read_folder)
    save_meta_folder = 'parking-pos-meta'
    if not os.path.exists(save_meta_folder):
        os.makedirs(save_meta_folder)
    save_review_folder = 'parking-pos-review'
    if not os.path.exists(save_review_folder):
        os.makedirs(save_review_folder)

    meta_files = sorted([filename for filename in filenames if filename.lower().startswith('meta')])
    review_files = sorted([filename for filename in filenames if filename.lower().startswith('review')])

    for state in us_states:
        for (filename1, filename2) in zip(meta_files, review_files):
            if (state.lower() in filename1.lower()) and (state.lower() in filename2.lower()):
                read_filename1 = os.path.join(read_folder, filename1)
                read_filename2 = os.path.join(read_folder, filename2)
                filtered_df1, filtered_df2 = filter_data(read_filename1, read_filename2)
                write_data(filtered_df1, filename1, filtered_df2, filename2)