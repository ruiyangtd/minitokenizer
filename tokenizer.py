class Tokenizer:
    def __init__(self, vocab_size:int):
        self._vocab_size = vocab_size
        self._vocab = dict()
        self._merge = dict()
        self._id2token = dict()

    def encode(self, sequence:str):
        words = list(sequence)

        while len(words) >= 2:
            pair2ind = {}
            for w0, w1 in zip(words[:-1], words[1:]):
                pair2ind[(w0, w1)] = self._merge.get((w0, w1), float('inf'))
            pair = min(pair2ind, key=pair2ind.get)
            if pair not in self._merge:
                break
            new_words = []
            j = 0
            while j < len(words):
                if j+1 < len(words) and words[j] == pair[0] and words[j+1] == pair[1]:
                    new_words.append(words[j]+words[j+1])
                    j += 2
                else:
                    new_words.append(words[j])
                    j += 1
            words = new_words
        return [self._vocab.get(word, "<unk>") for word in words]
        


    def encode_batch(self, sequences:list):
        pass

    def decode(self, ids:list):
        words = []
        for id in ids:
            words.append(self._id2token.get(id, ''))
        return words

    def decode_batch(self, sequences:list):
        pass

    def train(self, text:str):
        # split
        words = list(text)
        for i in range(self._vocab_size):
            # count pair
            pair2count = {}
            for w0, w1 in zip(words[:-1], words[1:]):
                pair = (w0, w1)
                pair2count[pair] = pair2count.get(pair, 0) + 1
            pair = max(pair2count, key=pair2count.get)
            # add token to vocab
            self._merge[pair] = i
            self._vocab[pair[0]+pair[1]] = i
            self._id2token[i] = pair[0]+pair[1]
            new_words = []
            j = 0
            while j < len(words):
                if j+1 < len(words) and words[j] == pair[0] and words[j+1] == pair[1]:
                    new_words.append(words[j]+words[j+1])
                    j += 2
                else:
                    new_words.append(words[j])
                    j += 1
            words = new_words

if __name__ == "__main__":
    tokenizer = Tokenizer(vocab_size=30)
    text = """我说道，“爸爸，你走吧。”他望车外看了看，说，“我买几个橘子去。
    你就在此地，不要走动。”我看那边月台的栅栏外有几个卖东西的等着顾客。
    走到那边月台，须穿过铁道，须跳下去又爬上去。父亲是一个胖子，走过去自然要费事些。
    我本来要去的，他不肯，只好让他去。我看见他戴着黑布小帽，穿着黑布大马褂，
    深青布棉袍，蹒跚地走到铁道边，慢慢探身下去，尚不大难。可是他穿过铁道，
    要爬上那边月台，就不容易了。他用两手攀着上面，两脚再向上缩；他肥胖的身子向左微倾，
    显出努力的样子。这时我看见他的背影，我的泪很快地流下来了。我赶紧拭干了泪，
    怕他看见，也怕别人看见。我再向外看时，他已抱了朱红的橘子望回走了。
    过铁道时，他先将橘子散放在地上，自己慢慢爬下，再抱起橘子
    走。到这边时，我赶紧去搀他。他和我走到车上，将橘子一股脑儿放在我
    的皮大衣上。于是扑扑衣上的泥土，心里很轻松似的，过一会说，“我走
    了；到那边来信！”我望着他走出去。他走了几步，回过头看见我，
    说，“进去吧，里边没人。”等他的背影混入来来往往的人里，再找不着
    了，我便进来坐下，我的眼泪又来了。
    """
    tokenizer.train(text=text)
    print(tokenizer._vocab)
    ids = tokenizer.encode(sequence="我看见铁道")
    text = tokenizer.decode(ids=ids)
    print(ids)
    print(text)




