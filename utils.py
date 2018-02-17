# Useful functions

import os

DATA_DIR = 'Gutenberg'

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

# data should already be organized using organize_by_author
def list_author_sizes():
    sizes = []
    for author in os.listdir(DATA_DIR):
        if author[0] != '.':
            author_dir = os.path.join(DATA_DIR, author)
            size = sum(os.path.getsize(os.path.join(author_dir, f))
                                       for f in os.listdir(author_dir) if
                                       os.path.isfile(os.path.join(author_dir,
                                                                   f)))
            print('%s: %d' % (author, size))
            sizes.append((author, size))
    print(sorted(sizes, key=lambda x: x[1]))
