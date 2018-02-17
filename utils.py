# Useful functions

import os

DATA_DIR = 'Gutenberg'

def organize_by_author():
    files = os.listdir(DATA_DIR)
    authors = set()
    for f in files:
        if f[0] != '.' and os.path.splitext(f)[1] == '.txt':
            author = f[:f.find('_')]
            if author not in authors:
                authors.add(author)
                if not os.path.exists(os.path.join(DATA_DIR, author)):
                    os.mkdir(os.path.join(DATA_DIR, author))
                print author
            os.rename(os.path.join(DATA_DIR, f),
                      os.path.join(DATA_DIR, author, f))
