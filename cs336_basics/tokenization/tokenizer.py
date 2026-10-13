import functools
import pickle
import numpy as np
import regex as re
from torch.multiprocessing import Pool

from cs336_basics.tokenization.utils import (
    PATTERN,
    get_merge_indices,
    merge_in_byte_word,
    merge_in_indices,
)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.merges = merges
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = [
                special_token.encode("utf-8") for special_token in special_tokens
            ]

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)

        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)

        return Tokenizer(vocab, merges, special_tokens)

    def _split_text(self, text) -> list[str]:
        if not self.special_tokens:
            return [text]
        docs = []
        pos = 0
        while pos < len(text):
            found_at_list = [
                (text.find(split_special_token, pos), split_special_token)
                for split_special_token in self.special_tokens
            ]
            found_at, found_special_token = min(
                found_at_list,
                key=lambda x: np.inf if x[0] < 0 else x[0] + 1 / (len(x[1]) + 0.001),
            )
            if found_at != -1:
                if pos != found_at:
                    docs.append(text[pos:found_at])
                pos = found_at + len(found_special_token)
                docs.append(text[found_at:pos])
            else:
                docs.append(text[pos:])
                break
        return docs

    def encode(self, text) -> list:
        if isinstance(text, str):
            text = bytes(text, encoding="utf-8")
        tokens_list = []
        docs = self._split_text(text)
        processed = 0

        cache = dict()
        for doc in docs:
            processed += 1

            percentage = 100 * processed / len(docs)
            print(
                f"\rProcessed: {percentage:5.2f}%",
                end="",
                flush=True,
            )
            if doc in self.special_tokens:
                tokens_list.extend([self.inverse_vocab.get(doc, 1)])
            else:
                for match in re.finditer(PATTERN, doc):
                    match_text = match.group()
                    if match_text in cache:
                        tokens = cache[match_text]
                    else:
                        pretokens = tuple([l.to_bytes() for l in match_text])

                        for merge in self.merges:
                            pretokens = merge_in_byte_word(merge, pretokens)

                        # pretokens_pairs = [(pretokens[i : i + 2]) for i in range(len(pretokens) - 1)]
                        # for merge in self.merges:
                        #     merge_indices = get_merge_indices(merge, pretokens_pairs)
                        #     if merge_indices is not None:
                        #         pretokens = merge_in_indices(merge, pretokens, merge_indices)
                        tokens = [self.inverse_vocab.get(bt, 1) for bt in pretokens]
                        cache[match_text] = tokens
                    tokens_list.extend(tokens)
        del cache
        # return [self.inverse_vocab.get(bt, 1) for bt in tokens_list]
        return tokens_list

    def encode_single(self, doc) -> list:
        tokens_list = []
        if doc in self.special_tokens:
            tokens_list.append((doc,))
        else:
            for match in re.finditer(PATTERN, doc):
                pretokens = tuple([l.to_bytes() for l in match.group()])
                pretokens_pairs = [
                    (pretokens[i : i + 2]) for i in range(len(pretokens) - 1)
                ]
                for merge in self.merges:
                    merge_indices = get_merge_indices(merge, pretokens_pairs)
                    if merge_indices is not None:
                        pretokens = merge_in_indices(merge, pretokens, merge_indices)
                tokens_list.append(pretokens)
        byte_tokens = [t for tokens in tokens_list for t in tokens]
        return [self.inverse_vocab.get(bt, 1) for bt in byte_tokens]

    def encode_parallel(self, text, num_processes=50) -> list:
        if isinstance(text, str):
            text = bytes(text, encoding="utf-8")
        docs = self._split_text(text)

        tokens_list = []

        with Pool(num_processes) as p:
            for processed, tokens in enumerate(
                p.imap(self.encode_single, docs),
                start=1,
            ):
                tokens_list.extend(tokens)

                percentage = 100 * processed / len(docs)
                print(
                    f"\rProcessed: {percentage:5.2f}%",
                    end="",
                    flush=True,
                )
        return tokens_list

    def encode_iterable(self, iterable):
        for text in iterable:
            for token in self.encode(text):
                yield token

    def decode(self, tokens):
        text = b""
        for token in tokens:
            text += self.vocab.get(token)
        return text.decode(errors="replace")


# SPECIAL_TOKENS = ["<|endoftext|>"]
# DEFAULT_VOCAB_LEN = len(SPECIAL_TOKENS) + 256
# if __name__ == "__main__":
#     # vocab, merges = tokenize("data/test.txt", DEFAULT_VOCAB_LEN + 6, SPECIAL_TOKENS)
#     tokenizer = Tokenizer.from_files("tokenizer/vocab.pkl", "tokenizer/merges.pkl", SPECIAL_TOKENS)
#     with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
#         t = f.read()[:100000]
#         tokens = tokenizer.encode(t)
#         tokens2 = tokenizer.encode_parallel(t, 4)
#     print(tokens)
#     result = tokenizer.decode(tokens)
#     pass
#     # vocab, merges = tokenize(
#     #     "data/TinyStoriesV2-GPT4-train.txt", 10_000, SPECIAL_TOKENS
#     # )

#     # vocab, merges = tokenize(
#     #     "data/owt_train.txt", 32_000, SPECIAL_TOKENS
#     # )
