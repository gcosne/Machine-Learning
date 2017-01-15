import os
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from nltk.collocations import FreqDist
from bs4 import BeautifulSoup
import string
import word2vec
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path = "data"
vectors_path = "word2vec/vectors.bin"
vocab_path = "word2vec/vocab.txt"

# "watermelon", "kumquat", "tangerine"
fruits = ["apple", "orange", "pear", "peach", "cherry", "pineapple", "plum", "strawberry", "apricot", "date", "fig",
          "grape", "raisin", "kiwi", "persimmon", "papaya", "mango", "tomato", "plum", "lemon"]
# "donut", "eclair", "popsicle",  "biscuit", "bubblegum", "gummy bear",
# "macaroon", "lollipop", "taffy", "wafer", "strudel"
junk = ["soda", "popcorn", "candy", "chocolate", "juice", "pretzel", "hamburger", "fries",
        "chips", "pie", "cake", "gum", "licorice", "cookies", "beer", "pizza"]


def unigram_frequencies():
    tokens = []
    for filename in os.listdir(path):
        with open(path + "/" + filename, 'r', encoding='utf-8', errors='ignore') as inputfile:
            current = ""
            for line in inputfile:
                current += line.lower().replace('\n', ' ')
            soup = BeautifulSoup(current, 'html.parser')
            tokens.extend([i for i in WordPunctTokenizer().tokenize(soup.get_text()) if i not in get_punctuation()])

    """
            8 files
            Words: 2230407
        """
    # print(len(tokens))
    fdist = FreqDist(tokens)

    # Unigram frequencies of the top 50 words in the table
    # [('the', 111125), ('to', 48908), ('of', 48156), ('a', 43364), ('and', 42282), ('in', 40759), ('s', 24542),
    # ('for', 20143), ('that', 17942), ('by', 15288), ('is', 14950), ('said', 14755), ('on', 14502), ('ur', 12512),
    # ('with', 12254), ('it', 12012), ('at', 11645), ('md', 10964), ('was', 10611), ('he', 10317), ('as', 10200),
    # ('lr', 10037), ('from', 9856), ('reg', 9213), ('bc', 9184), ('ql', 9035), ('new', 7635), ('be', 7582),
    # ('are', 7575), ('has', 7412), ('his', 7341), ('have', 7225), ('an', 6693), ('1', 6646), ('but', 6611),
    # ('i', 6352), ('will', 5981), ('who', 5688), ('they', 5662), ('this', 5654), ('or', 5638), ('not', 5627),
    # ('tl', 5580), ('qc', 5177), ('percent', 5036), ('year', 4994), ('its', 4843), ('news', 4778), ('t', 4634),
    # ('more', 4627)]
    # print(fdist.most_common(50))

    """
            Fruits

            apple 152
            orange 44
            pear 7
            peach 7
            cherry 35
            pineapple 4
            plum 11
            strawberry 4
            apricot 6
            date 161
            fig 8
            grape 1
            raisin 2
            kiwi 4
            persimmon 16
            papaya 2
            mango 9
            tomato 8
            plum 11
            lemon 17
        """
    print("\nFruits")
    for item in fruits:
        print(item, fdist[item])

    """
        Junk Food

        soda 25
        popcorn 4
        candy 14
        chocolate 20
        juice 60
        pretzel 4
        hamburger 12
        fries 9
        chips 100
        pie 9
        cake 27
        gum 6
        licorice 1
        cookies 3
        beer 61
        pizza 48
    """
    print("\nJunk Food")
    for item in junk:
        print(item, fdist[item])


def create_word_vectors():
    word2vec.word2vec(train="word2vec/combined.txt", output=vectors_path,
                      save_vocab=vocab_path,
                      verbose=True, size=100, threads=4, min_count=1)


def word_vectors():
    model = word2vec.load(vectors_path)
    print("model.vocab:", len(model.vocab))
    print("model.vectors.shape:", model.vectors.shape)

    # for item in fruits:
    #     if item not in model:
    #         print("\t", item, "not found in model.")
    #     else:
    #         print("Found", item)
    #
    # for item in junk:
    #     if item not in model:
    #         print("\t", item, "not found in model.")
    #     else:
    #         print("Found", item)

    """
        model['apple'].shape: (100,)
        model['apple']: [ 0.17030741  0.08428216  0.06059612 -0.00638814  0.10112851 -0.1241515
          0.06725635  0.02879349 -0.04809854  0.22880751  0.01358577  0.15691407
         -0.13732363 -0.17997995 -0.02058961  0.0727186   0.21973862  0.07463118
          0.04177584  0.17932948 -0.13436042  0.11937297  0.15230356 -0.18795085
          0.05100906 -0.05926591  0.02826262 -0.06584113  0.0558053  -0.04116041
         -0.00987223  0.09669606 -0.0868069   0.11637654 -0.08530871 -0.09909809
          0.13360481 -0.0743489  -0.01872801 -0.1187283  -0.02732293 -0.10965312
         -0.05026517  0.06576528  0.08707665  0.14344434  0.09168278 -0.07285824
         -0.0310544  -0.13116072  0.02581204 -0.10003031 -0.05911924  0.10481259
         -0.14427128 -0.0633673   0.0107077  -0.05672868  0.08278223 -0.22492008
         -0.11084548  0.0513422   0.07521619 -0.00662204 -0.0160759   0.12412996
         -0.02550386 -0.09733595  0.18104105  0.13598791  0.09548864  0.06177835
          0.05140171 -0.08365466 -0.08046116 -0.05396777 -0.04628954  0.07251068
         -0.07222441  0.03830035  0.13375503  0.00371481  0.13437654  0.11950362
          0.04434878  0.09584519 -0.07209467 -0.11722135  0.05843765 -0.04014708
          0.11419578  0.00483528  0.01424075  0.06072676  0.07384299 -0.01508367
          0.08354376  0.15720205 -0.09291109 -0.1737577 ]

        model['pizza'].shape: (100,)
        model['pizza']: [ 0.00972527  0.05202191 -0.00700272 -0.17324953  0.15565492 -0.01572789
          0.07770704  0.05030056 -0.01214633  0.24957703  0.12860766  0.14831014
         -0.14189921 -0.11704075  0.04517415  0.05690712  0.26092535  0.10482656
          0.03936176  0.15667203 -0.14656599  0.06268185  0.13598818 -0.25435922
          0.02601384 -0.06412815 -0.00808344  0.0183344   0.05106594 -0.05716141
          0.05806965  0.05378154 -0.06747798  0.07269593 -0.02390247 -0.03033183
          0.14211419 -0.07691243  0.02733652 -0.18312091  0.00522892 -0.10024268
         -0.15034939 -0.04846548  0.12511389  0.12529902  0.03826363 -0.10610558
         -0.08233915 -0.15671478 -0.0042906  -0.17803754  0.06344658  0.06485491
         -0.04543499 -0.04426112  0.09156568 -0.05210818  0.13277449 -0.10320026
         -0.00916333 -0.10520437  0.08760599 -0.00371471 -0.05246779  0.20324129
          0.01605334 -0.11632558  0.15277274  0.15020384  0.00422594  0.05790341
          0.11455409 -0.04918056 -0.05096141  0.0059245  -0.00423472  0.01345381
         -0.18383133 -0.02310799  0.00082699 -0.01629616 -0.01088699  0.03178773
          0.10518334  0.14584099 -0.10157438 -0.06920595  0.052466   -0.08443438
          0.02989486 -0.05647052  0.05308264 -0.06343999  0.03199876  0.09277579
          0.03335597  0.11092914 -0.04178311 -0.1558584 ]
    """
    # print("model['apple'].shape:", model['apple'].shape)
    # print("model['apple']:", model['apple'])
    # print("model['pizza'].shape:", model['pizza'].shape)
    # print("model['pizza']:", model['pizza'])


def pca():
    """
        PCA example
        http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#drop_labels

        :return:
    """
    model = word2vec.load(vectors_path)

    vect_2d = []
    for item in fruits:
        vect_2d.append(model[item])

    for item in junk:
        vect_2d.append(model[item])

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(vect_2d)

    plt.figure(1)
    plt.plot(transformed[0:len(fruits), 0], transformed[0:len(fruits), 1], 'o', markersize=7, color='blue',
             label='fruit')
    plt.plot(transformed[len(fruits):len(vect_2d), 0], transformed[len(fruits):len(vect_2d), 1], '^', markersize=7,
             color='red', label='junk')
    plt.legend()
    plt.show()


def kmeans():
    model = word2vec.load(vectors_path)

    vect_2d = []
    for item in fruits:
        vect_2d.append(model[item])

    for item in junk:
        vect_2d.append(model[item])

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(vect_2d)

    clusters = KMeans(n_clusters=5).fit(transformed)

    plt.figure(2)
    plt.plot(transformed[0:len(fruits), 0], transformed[0:len(fruits), 1], 'o', markersize=7, color='blue',
             label='fruit')
    plt.plot(transformed[len(fruits):len(vect_2d), 0], transformed[len(fruits):len(vect_2d), 1], '^', markersize=7,
             color='red', label='junk')
    plt.plot(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], '8', markersize=10, color='green',
             label='cluster center')
    plt.legend()
    plt.show()


def consolidate_text():
    with open("word2vec/combined.txt", 'w') as outputfile:
        for filename in os.listdir(path):
            with open(path + "/" + filename, 'r', encoding='utf-8', errors='ignore') as inputfile:
                current = ""
                for line in inputfile:
                    current += line.lower().replace('\n', ' ')
                soup = BeautifulSoup(current, 'html.parser')
                outputfile.write(soup.get_text())


def get_punctuation():
    punctuations = list(string.punctuation)
    extra_punctuation = ["``", ",''", ".''"]
    punctuations.extend(extra_punctuation)
    return punctuations


if __name__ == "__main__":
    # unigram_frequencies()
    # consolidate_text()  # helper method
    # create_word_vectors()
    # word_vectors()
    # pca()
    kmeans()
