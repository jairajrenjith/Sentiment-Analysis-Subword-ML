import json
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

    def get_stats(self, tokens):
        pairs = defaultdict(int)
        for word, freq in tokens.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, tokens):
        new_tokens = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in tokens:
            new_tokens[word.replace(bigram, replacement)] = tokens[word]
        return new_tokens

    def train(self, texts):
        tokens = Counter()
        for text in texts:
            for word in text.lower().split():
                tokens[' '.join(word) + ' </w>'] += 1

        for _ in range(self.vocab_size):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            tokens = self.merge_vocab(best, tokens)

        vocab = set()
        for word in tokens:
            vocab.update(word.split())

        self.vocab = {t: i for i, t in enumerate(vocab)}

    def tokenize(self, text):
        output = []
        for word in text.lower().split():
            chars = list(word) + ['</w>']
            i = 0
            while i < len(chars):
                j = len(chars)
                while j > i and ''.join(chars[i:j]) not in self.vocab:
                    j -= 1
                output.append(''.join(chars[i:j]))
                i = j
        return output

    def save_vocab(self, path):
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)
