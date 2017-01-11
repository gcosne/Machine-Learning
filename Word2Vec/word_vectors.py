import os
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from nltk.collocations import FreqDist
from bs4 import BeautifulSoup
import string
from gensim.models import word2vec

if __name__ == "__main__":
    tokens = []
    sentences = []
    # read in data
    punctuations = list(string.punctuation)
    extra_punctuation = ["``", ",''", ".''"]
    punctuations.extend(extra_punctuation)
    path = "data"
    for filename in os.listdir(path):
        with open("data/" + filename, 'r') as inputfile:
            current = ""
            for line in inputfile:
                current += line.lower().replace('\n', ' ')
            soup = BeautifulSoup(current, 'html.parser')
            # tokens.extend(WordPunctTokenizer().tokenize(soup.get_text()))
            # tokens.extend([i for i in WordPunctTokenizer().tokenize(soup.get_text()) if i not in punctuations])
            sentences.append(sent_tokenize(soup.get_text()))

    """
        8 files
        Words: 2230407
    """
    # print(len(tokens))
    # fdist = FreqDist(tokens)

    # Unigram frequencies of the top 50 words in the table
    # [('the', 111125), ('to', 48908), ('of', 48156), ('a', 43364), ('and', 42282), ('in', 40759), ('s', 24542), ('for', 20143), ('that', 17942), ('by', 15288), ('is', 14950), ('said', 14755), ('on', 14502), ('ur', 12512), ('with', 12254), ('it', 12012), ('at', 11645), ('md', 10964), ('was', 10611), ('he', 10317), ('as', 10200), ('lr', 10037), ('from', 9856), ('reg', 9213), ('bc', 9184), ('ql', 9035), ('new', 7635), ('be', 7582), ('are', 7575), ('has', 7412), ('his', 7341), ('have', 7225), ('an', 6693), ('1', 6646), ('but', 6611), ('i', 6352), ('will', 5981), ('who', 5688), ('they', 5662), ('this', 5654), ('or', 5638), ('not', 5627), ('tl', 5580), ('qc', 5177), ('percent', 5036), ('year', 4994), ('its', 4843), ('news', 4778), ('t', 4634), ('more', 4627)]
    # print(fdist.most_common(50))

    # "watermelon", "kumquat",
    fruits = ["apple", "orange", "pear", "peach", "cherry", "pineapple", "plum", "strawberry", "apricot", "date", "fig",
              "grape", "raisin", "kiwi", "persimmon", "tangerine", "mango", "tomato", "plum", "lemon"]
    # "donut", "eclair", "popsicle",  "biscuit", "bubblegum", "gummy bear", "macaroon", "lollipop", "taffy"
    junk = ["soda", "popcorn", "candy", "wafer", "chocolate", "juice", "strudel", "pretzel", "hamburger", "fries",
            "chips", "pie", "cake", "gum", "licorice", "cookies"]

    """
        Fruits
        - - - - - - -
        apple       152
        orange      44
        pear        7
        peach       7
        cherry      35
        pineapple   4
        plum        11
        strawberry  4
        apricot     6
        date        161
        fig         8
        grape       1
        raisin      2
        kiwi        4
        persimmon   16
        tangerine   2
        mango       9
        tomato      8
        plum        11
        lemon       17
    """
    # print("\nFruits")
    # for item in fruits:
    #     print(item, fdist[item])

    """
        Junk Food
        soda        25
        popcorn     4
        candy       14
        wafer       1
        chocolate   20
        juice       60
        strudel     1
        pretzel     4
        hamburger   12
        fries       9
        chips       100
        pie         9
        cake        27
        gum         6
        licorice    1
        cookies     3
    """
    # print("\nJunk Food")
    # for item in junk:
    #     print(item, fdist[item])


    # for filename in os.listdir(path):
    #     with open("data/" + filename, 'r') as inputfile:
    #         soup = BeautifulSoup(inputfile.read(), 'html.parser')
    #         print(sent_tokenize(soup.get_text()))

    print(sentences)
    word2vec(sentences)
