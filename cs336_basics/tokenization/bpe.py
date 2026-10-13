import cProfile
from collections import Counter, defaultdict
from functools import reduce
import operator
import pickle
import time
import regex as re

from cs336_basics.tokenization.utils import (
    PATTERN,
    merge_pair,
    process_file,
    merge_in_byte_word,
)


def pretokenize_serial(text: str) -> dict:
    word_counter = Counter()
    for match in re.finditer(PATTERN, text):
        word_counter[tuple([l.to_bytes() for l in match.group()])] += 1

    return word_counter


def pretokenize(input_path, special_tokens) -> dict:
    pretokenized_docs = process_file(input_path, special_tokens, pretokenize_serial, 48)
    return reduce(operator.add, pretokenized_docs)


def count_pairs(byte_word_counter):
    pair_counter = Counter()
    pair_index = defaultdict(set)
    for byte_word, count in byte_word_counter.items():
        for t_curr, t_next in zip(byte_word, byte_word[1:]):
            pair_counter[(t_curr, t_next)] += count
            pair_index[(t_curr, t_next)].add(byte_word)
    return pair_counter, pair_index


def update_count_pairs(pair_counter, pair_index, byte_word_counter, new_token_pair):
    curr_index = pair_index[new_token_pair].copy()
    for old_byte_word in curr_index:
        new_byte_word = merge_in_byte_word(new_token_pair, old_byte_word)
        curr_count = byte_word_counter[old_byte_word]
        del byte_word_counter[old_byte_word]
        byte_word_counter[new_byte_word] = curr_count

        for t_curr, t_next in zip(old_byte_word, old_byte_word[1:]):
            pair_counter[(t_curr, t_next)] -= curr_count
            if old_byte_word in pair_index[(t_curr, t_next)]:
                pair_index[(t_curr, t_next)].remove(old_byte_word)

        for t_curr, t_next in zip(new_byte_word, new_byte_word[1:]):
            pair_counter[(t_curr, t_next)] += curr_count
            pair_index[(t_curr, t_next)].add(new_byte_word)

    del pair_counter[new_token_pair]
    del pair_index[new_token_pair]

    return pair_counter, pair_index, byte_word_counter


def update_byte_word_counter(byte_word_counter, new_token_pair):
    byte_words = list(byte_word_counter.keys())
    for byte_word in byte_words:
        new_byte_word = merge_in_byte_word(new_token_pair, byte_word)
        byte_word_counter[new_byte_word] = byte_word_counter.pop(byte_word)
    return byte_word_counter


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = [special_token.encode("utf-8") for special_token in special_tokens]
    vocab = special_tokens + [bytes([i]) for i in range(256)]
    merges = []
    byte_word_counter = pretokenize(input_path, special_tokens)

    pair_counter, pair_index = count_pairs(byte_word_counter)
    while len(vocab) < vocab_size:
        most_common_num = max(pair_counter.values())
        new_token_pair = sorted(
            [key for key in pair_counter if pair_counter[key] == most_common_num]
        )[-1]
        merges.append(new_token_pair)
        new_token = merge_pair(new_token_pair)
        vocab.append(new_token)

        pair_counter, pair_index, byte_word_counter = update_count_pairs(
            pair_counter, pair_index, byte_word_counter, new_token_pair
        )
    return dict(enumerate(vocab)), merges


if __name__ == "__main__":
    SPECIAL_TOKENS = ["<|endoftext|>"]
    DEFAULT_VOCAB_LEN = len(SPECIAL_TOKENS) + 256
    # cProfile.run(
    #     "train_bpe('data/TinyStoriesV2-GPT4-train.txt', 1000, SPECIAL_TOKENS)",
    #     sort="cumtime"
    # )
    start = time.time()
    vocab, merges = train_bpe("data/test.txt", DEFAULT_VOCAB_LEN + 6, SPECIAL_TOKENS)
    # vocab, merges = train_bpe(
    #     "data/TinyStoriesV2-GPT4-train.txt", 10_000, SPECIAL_TOKENS
    # )
    vocab, merges = train_bpe("data/owt_train.txt", 32_000, SPECIAL_TOKENS)
    end = time.time()
    # print("{:.4f} mins".format((end - start)/60.0))

    with open("storage/tokenizers/owt/vocab.pkl", "wb+") as vf:
        pickle.dump(vocab, vf)

    with open("storage/tokenizers/owt/merges.pkl", "wb+") as mf:
        pickle.dump(merges, mf)

    print(vocab)
    print(merges)
