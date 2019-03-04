#encoding:utf-8
import numpy as np
def load_tencent_word_embedding(path):
    words = {}
    fp = open(path, 'r')
    line = fp.readline()
    count = 0
    
    while True:
        line = fp.readline()
        word, floats =  line.split(' ',1)
        words[word] = np.fromstring(floats, dtype=np.float, count=200, sep=' ')
        count += 1
    return words

    
    
if __name__ == '__main__':
    words = load_tencent_word_embedding('../var/Tencent_AILab_ChineseEmbedding.txt')
    print(words)
    

