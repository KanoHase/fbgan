'''
import MeCab
wakati = MeCab.Tagger("-Owakati")
words = wakati.parse("ここではきものを脱いでください").split()
print(words, type(words)) # list
'''

from gensim.models import Word2Vec
from implementations.data_utils import prepare_binary
from torchtext.vocab import Vectors

data_dir = "./data/"
binary_positive_data_file = "amino_positive_541.txt"
binary_negative_data_file = "amino_negative_541.txt"
neg = True

seq_arr, _, _ = prepare_binary(data_dir + binary_positive_data_file, data_dir + binary_negative_data_file, neg)
print(seq_arr)

# インスタンスの生成
model = Word2Vec(seq_arr,size=10,window=1)
model.train(seq_arr,total_examples=len(seq_arr),epochs=100)

print(model.wv["C"])
print(model.wv.most_similar("D"))
print(model.wv)

model.wv.save_word2vec_format('./data/amino_word2vec_vectors.vec')
amino_vec = Vectors(name='./data/amino_word2vec_vectors.vec')
print("1単語を表現する次元数：", amino_vec.dim)
print("単語数：", len(amino_vec.itos))


'''
# wakati_file = "pos_wakati.txt"

#seq_arr_period = []

def make_wakati_file(pos_path, gram=1):
    with open(pos_path) as f:
        for line in f:
            tmp = list(line[:-1])
            tmp.append(".")
            seq_arr_period.append(tmp)
    return seq_arr_period

# seq_arr_period = make_wakati_file(data_dir+binary_positive_data_file)
# print(seq_arr_period)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('./wiki_wakati.txt')

model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)
model.wv.save_word2vec_format("./wiki.vec.pt", binary=True)
'''