from gensim.models import KeyedVectors

if __name__ == "__main__":
    print("loading...")
    model = KeyedVectors.load_word2vec_format("./enwiki_20180420_100d.txt", binary=False)
    print("saving...")
    model.save("enwiki_20180420_100d.bin")
    print("Done!")
