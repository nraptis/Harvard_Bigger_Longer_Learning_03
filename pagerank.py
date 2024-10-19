import os
import random
import re
import sys

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

#
#
# should return a dictionary representing the probability distribution over which page a random surfer would visit next, given a corpus of pages, a current page, and a damping factor.
#
#The return value of the function should be a Python dictionary with one key for each page in the corpus. Each key should be mapped to a value representing the probability that a random surfer would choose that page next. The values in this returned probability distribution should sum to 1.
#
#

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
    
    pages = all_pages(corpus)

    result = {}

    for page in pages:
        result[page] = 0.0

    transition_model_pages = {}
    transition_model_weights = {}
    for page in pages:
        _transition_model = transition_model(corpus, page, damping_factor)
        transition_model_pages[page] = list(_transition_model.keys())
        transition_model_weights[page] = list(_transition_model.values())
        
    current_page = random.choice(pages)

    for _ in range(n):
        result[current_page] += 1
        _pages = transition_model_pages[current_page]
        _weights = transition_model_weights[current_page]
        current_page = random.choices(_pages, weights = _weights, k = 1)[0]

    for page in pages:
        result[page] /= float(n)

    return result

'''
def sample_pagerank(corpus, damping_factor, n):

    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pages = list(corpus.keys())
    result = {}
    for page in pages:
        result[page] = 0.0

    odds = transition_model(corpus, page, damping_factor)
    page = random.choice(pages)
    for _ in range(0, n):
        page_list = random.choices(list(odds.keys()), list(odds.values()))
        page = page_list[0]
        result[page] += 1.0
        odds = transition_model(corpus, page, damping_factor)

    for page in pages:
        result[page] /= n

    return result
'''

'''
def sample_pagerank(corpus, damping_factor, n):

    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    result = {}

    for page in corpus:
        result[page] = 0.0

    pages = all_pages(corpus)

    starting_page = random.choice(pages)
    for _ in range(n):
        options = []
        weights = []

        probabilities = transition_model(corpus, starting_page, damping_factor)

        for page in probabilities:
            options.append(page)
            weights.append(probabilities[page])
    
        starting_page = random.choices(options, weights = weights)[0]
        result[starting_page] += 1

    for page in result:
        result[page] /= n

    return result
'''



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    backward_corpus = {}

    for page in corpus:
        if not corpus[page]:
            for x in corpus:
                corpus[page].add(x)


    for page in corpus:
        backward_corpus[page] = set()

    for page in corpus:
        for linked_page in corpus[page]:
            backward_corpus[linked_page].add(page)

    page_ranks = {}
    num_pages = len(corpus)
    end = False

    for page in corpus:
        page_ranks[page] = 1 / num_pages
    
    while not end:
        temp_page_ranks = dict(page_ranks)

        for page in page_ranks:
            probability_sum = 0
            for linked_page in backward_corpus[page]:
                probability_sum += page_ranks[linked_page] / len(corpus[linked_page])
            page_ranks[page] = ((1 - damping_factor) / num_pages) + (damping_factor * probability_sum)
            
        end = True
        for page in page_ranks:
            if not temp_page_ranks[page] * 0.999 < page_ranks[page] < temp_page_ranks[page] * 1.001:
                end = False

    return page_ranks

if __name__ == "__main__":
    main()
