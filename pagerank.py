import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

def all_pages(corpus):
    result = set()
    for key in corpus.keys():
        result.add(key)
        for value in corpus[key]:
                result.add(value)
    return list(result)

def transition_model(corpus, page, damping_factor):

    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    pages_all = all_pages(corpus)
    pages_all_count = len(pages_all)
    
    pages_links = corpus[page]
    pages_links_count = len(pages_links)

    result = {}

    damping_factor_inverse = (1.0 - damping_factor)
    
    #these sum up to damping_factor_inverse
    for page in pages_all:
        result[page] = damping_factor_inverse / pages_all_count

    #these sum up to damping_factor
    for page in pages_links:
        result[page] += damping_factor / pages_links_count

    #these sum up to 1.0 (damping_factor_inverse + damping_factor)
    return result

def sample_pagerank(corpus, damping_factor, n):

    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pages_all = all_pages(corpus)
    pages_all_count = len(pages_all)

    result = {}

    for page in pages_all:
        result[page] = 0.0

    transition_model_pages = {}
    transition_model_weights = {}
    for page in pages_all:
        _transition_model = transition_model(corpus, page, damping_factor)
        transition_model_pages[page] = list(_transition_model.keys())
        transition_model_weights[page] = list(_transition_model.values())
        
    current_page = random.choice(pages_all)

    for _ in range(n):
        result[current_page] += 1
        _pages = transition_model_pages[current_page]
        _weights = transition_model_weights[current_page]
        current_page = random.choices(_pages, weights = _weights, k = 1)[0]

    for page in pages_all:
        result[page] /= float(n)

    return result

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pages_all = all_pages(corpus)
    pages_all_count = len(pages_all)

    damping_factor_inverse = (1.0 - damping_factor)

    result = {}

    for page in pages_all:
        result[page] = 1.0 / pages_all_count

    reverse_corpus = {}

    for page in pages_all:
        reverse_corpus[page] = set()

    for page in pages_all:
        for link in corpus[page]:
            reverse_corpus[link].add(page)

    for page in reverse_corpus:
        reverse_corpus[page] = list(reverse_corpus[page])

    fudge = 100000
    for _ in range(0, fudge):
        out_of_range = False
        hold = dict(result)

        for page in pages_all:

            # The left-hand-side of the equation...
            rank_lhs = damping_factor_inverse / pages_all_count

            # The right-hand-side of the equation...
            rank_rhs = 0.0
            for link in reverse_corpus[page]:
                num_links = len(corpus[link])
                rank_rhs += result[link] / num_links
            rank_rhs *= damping_factor

            # The new estimated page rank.
            result[page] = rank_lhs + rank_rhs
            
        for page in pages_all:
            # If we are within epsilon, game over...
            if abs(hold[page] - result[page]) > 0.0001:
                out_of_range = True

        if not out_of_range:
            break

    return result


if __name__ == "__main__":
    main()
