import os
from os.path import join
import pickle

"""
This script is designed for one-time use. 
It will take the directories of authors, with files of books inside,
and reorganize it into one file of all the author's text.

After trimming, the file is split into consecutive chunks of 100 words,
and distributed into train/val/test with ratio 80/10/10 resp.
"""

DATA_DIR = "../data" # change this to be actual dir
OUTPUT_DIR = "../data_processed"
TRAIN_DIR = join(OUTPUT_DIR, "train")
VAL_DIR = join(OUTPUT_DIR, "val")
TEST_DIR = join(OUTPUT_DIR, "test")


TRIM_MARGIN = 500
THRESHOLD = 4 * TRIM_MARGIN
CHUNK_LENGTH = 150

def get_author_dirs():
    def is_author_name(dirname):
        for word in dirname.split(' '):
            if not word.isalpha():
                return False
        return True

    return [d for d in os.listdir(DATA_DIR) if is_author_name(d)]


def get_texts_in_dir(adir):
    author_text = []

    for file in [f for f in os.listdir(adir) if f[-4:] == '.txt']:
        with open(join(adir, file), 'r') as f:
            contents = f.read().split(' ')

        if len(contents) < THRESHOLD:
            print("File %s has < %d words. Skipping." % (file, THRESHOLD))

        contents = contents[TRIM_MARGIN:-TRIM_MARGIN]
        
        # Cut to multiple of 150. We don't want a chunk bridging two documents
        truncate_length = (len(contents) // CHUNK_LENGTH) * CHUNK_LENGTH
        contents = contents[:truncate_length]
        author_text.extend(contents)

    return author_text 

def distribute_into_output_dir(authorname, author_text):
    assert(len(author_text) % CHUNK_LENGTH == 0)
    
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(join(split, authorname))

    index = 0
    while len(author_text) > 0:
        
        # break off a chunk
        chunk = author_text[:CHUNK_LENGTH]
        author_text = author_text[CHUNK_LENGTH:]

        # Determine if it goes into train, dev, test
        diceroll = random.random()
        split_dir = TRAIN_DIR if diceroll < 0.8 else (VAL_DIR if diceroll < 0.9 else TEST_DIR)

        chunk_name = "%s__%010d.pkl" % (authorname, index) # have author, and index padded up to 10 places
        index += 1

        destination = join(split_dir, authorname, chunk_name)

        # put in destination under author with that name
        with open(destination, 'wb') as f:
            pickle.dump(chunk, f)

if __name__ == "__main__":
    author_dirs = get_author_dirs()
    for adir in author_dirs: # adir is the author name string
        author_text = get_texts_in_dir(adir) 
        distribute_into_output_dir(adir, author_text)
