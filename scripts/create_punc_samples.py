import os
from os.path import join, isdir
import pickle
import random

'''
This script is designed for one-time use.
It will take the directories of authors, with files of books inside,
and reorganize it into one file of all the author's text.

After trimming, we strip out all alphanumeric characters, leaving
punctuation only. Then the file is split into consecutive chunks of 500
punctuations, and distributed into train/val/test with ratio 80/10/10 resp.
'''

DATA_DIR = '../Gutenberg'  # change this to be actual dir
OUTPUT_DIR = '../data'
AUTHOR_LIST_FILE = 'unique_authors.txt'
TRAIN_DIR = join(OUTPUT_DIR, 'train')
VAL_DIR = join(OUTPUT_DIR, 'val')
TEST_DIR = join(OUTPUT_DIR, 'test')
PUNCS = set('.,?!\'":;-()')

THRESHOLD = 2000
CHUNK_LENGTH = 1000


def get_author_dirs(path=DATA_DIR):
    def is_author(name):
        return ''.join(name.split('_')).isalpha()
    print('Retrieving the authors under: %s' % path)
    alist = [d for d in os.listdir(path) if is_author(d)]
    print('We found authors: %s' % str(alist))
    return alist


def compress_tokens(l):
    punc_idx = [-1] + [i for i, x in enumerate(l) if x in PUNCS]
    return [(punc_idx[i] - punc_idx[i-1] - 1, l[punc_idx[i]]) for i in
            range(1, len(punc_idx))]


def strip_to_puncs(s):
    import re
    return re.findall(r'[\w]+|[\.,\?!\'":;-]|[(]|[)]', s)


def get_puncs_in_dir(adir):
    tokens = []  # list of punctuations from all texts under the author

    print('Getting puncutations from author %s.' % adir)
    author_files = [f for f in os.listdir(join(DATA_DIR, adir)) if f[-4:] ==
                    '.txt' and f[0] != '.']
    print('Found texts: %s' % author_files)

    for fil in author_files:
        with open(join(DATA_DIR, adir, fil), 'r') as f:
            contents = strip_to_puncs(f.read())

        print('length of content (# tokens) for %s, %d.' % (fil, len(contents)))
        if len(contents) < THRESHOLD:
            print('File %s has < %d tokens. Skipping.' % (fil, THRESHOLD))

        # Cut to multiple of CHUNK_LENGTH.
        # Don't want a chunk bridging two documents later
        truncate_length = (len(contents) // CHUNK_LENGTH) * CHUNK_LENGTH
        contents = contents[:truncate_length]
        tokens.extend(contents)

    print('Aggregated text has %d tokens (author %s).' % (len(tokens), adir))
    return tokens


def distribute_into_output_dir(authorname, tokens):
    assert(len(tokens) % CHUNK_LENGTH == 0)
    print('Distributing chunks into dir for %s' % authorname)

    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not isdir(join(split, authorname)):
            print('Made new dir for %s, split %s' % (authorname, split))
            os.makedirs(join(split, authorname))

    index = 0
    chop = 0
    while chop < len(tokens):

        # access a chunk
        chunk = tokens[chop:chop+CHUNK_LENGTH]
        compressed = compress_tokens(chunk)
        payload = ' '.join(['%d %s' % (x, y) for x, y in compressed])
        chop += CHUNK_LENGTH

        # Determine if it goes into train, dev, test
        diceroll = random.random()
        split_dir = TRAIN_DIR if diceroll < 0.8 else (VAL_DIR if diceroll < 0.9
                                                      else TEST_DIR)

        # have author, and index padded up to 10 places
        chunk_name = '%s__%010d.txt' % (authorname, index)

        destination = join(split_dir, authorname, chunk_name)

        # put in destination under author with that name
        with open(destination, 'wb') as f:
            pickle.dump(payload, f)

        index += 1
        if index % 100 == 0:
            print('At index %d...' % index)

    print('Done. total %d chunks (of %d tokens) created.' % (index,
                                                             CHUNK_LENGTH))


if __name__ == '__main__':
    author_dirs = get_author_dirs()
    # any split works (except train, removed from those for space)
    existing_adirs = get_author_dirs(path=TEST_DIR)

    remaining_adirs = set(author_dirs) - set(existing_adirs)
    print('The remaining authors are: %s' % str(remaining_adirs))
    errored_authors = []

    for adir in remaining_adirs:  # adir is the author name string
        # try:
            tokens = get_puncs_in_dir(adir)
            distribute_into_output_dir(adir, tokens)
            '''
        except Exception as e:
            print('Got error: %s' % str(e))
            print('Author %s may not be finished!' % adir)
            errored_authors.append(adir)
            '''
