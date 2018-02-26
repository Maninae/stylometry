# Useful functions

import os
import shutil

DATA_DIR = 'Gutenberg'
BACKUP_DIR = 'Gutenberg.BAK'

def organize_by_author():
    files = os.listdir(DATA_DIR)
    authors = set()
    for f in files:
        if f[0] != '.' and os.path.splitext(f)[1] == '.txt':
            author = f[:f.find('_')]
            author_dir = os.path.join(DATA_DIR, author)
            if author not in authors:
                authors.add(author)
                if not os.path.exists(author_dir):
                    os.mkdir(author_dir)
            os.rename(os.path.join(DATA_DIR, f),
                      os.path.join(author_dir, f))


def get_dir_size(DIR):
    return sum(os.path.getsize(os.path.join(DIR, f))
               for f in os.listdir(DIR) if
               os.path.isfile(os.path.join(DIR, f)))



# data should already be organized using organize_by_author
def list_author_sizes():
    sizes = []
    for author in os.listdir(DATA_DIR):
        if author[0] != '.':
            author_dir = os.path.join(DATA_DIR, author)
            size = get_dir_size(author_dir)
            sizes.append((author, size))
    for x, y in sorted(sizes, key=lambda x: x[1]):
        print x, y


def trim_authors(threshold=1000000):
    for author in os.listdir(DATA_DIR):
        if author[0] != '.':
            if get_dir_size(os.path.join(DATA_DIR, author)) < threshold:
                shutil.rmtree(os.path.join(DATA_DIR, author))


def remove_spaces():
    for author in os.listdir(DATA_DIR):
        if author[0] != '.':
            author_dir = os.path.join(DATA_DIR, author)
            for f in os.listdir(author_dir):
                if f[0] != '.':
                    os.rename(os.path.join(author_dir, f),
                              os.path.join(author_dir, f.replace(' ',
                                                                      '_')))
            os.rename(os.path.join(DATA_DIR, author),
                      os.path.join(DATA_DIR, author.replace(' ', '_')))
