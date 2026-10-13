import cProfile
from collections import Counter, defaultdict
from functools import reduce
import functools
import operator
import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool

PATTERN = (
    rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _split_text_file_in_chunks(
    file: BinaryIO,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(
        isinstance(split_special_token, bytes)
        for split_special_token in split_special_tokens
    ), "Must represent special tokens as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [0]

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    eof = False
    while not eof:
        initial_position = chunk_boundaries[-1]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries.append(file_size)
                eof = True
                break

            # Find the special token in the mini chunk
            found_at_list = [
                (mini_chunk.find(split_special_token), split_special_token)
                for split_special_token in split_special_tokens
            ]
            found_at, found_special_token = min(found_at_list, key=lambda x: x[0])
            if found_at != -1:
                chunk_boundaries.append(
                    initial_position + found_at + len(found_special_token)
                )
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def proccess_chunk(chunks, input_path, special_tokens, process_func):
    with open(input_path, "rb") as f:
        processed_docs = []
        for start, end in zip(chunks[:-1], chunks[1:]):
            f.seek(start)
            text = f.read(end - start)
            for special_token in special_tokens:
                text = text.replace(special_token, b"")
            processed_docs.append(process_func(text))
        return processed_docs


def process_file(input_path, special_tokens, process_func, num_processes=100):
    with open(input_path, "rb") as f:
        boundaries = _split_text_file_in_chunks(f, special_tokens)

    parts_size = len(boundaries) // num_processes
    chunks_list = []
    # for i in range(num_processes):
    #     start = max(i * parts_size - 1, 0)
    #     end = min((parts_size + 1) * i, len(boundaries))
    #     chunks_list.append(boundaries[start:end])
    #     if end == len(boundaries):
    #         break

    for i in range(num_processes):
        start = i * parts_size
        if i < num_processes - 1:
            end = (i+1) * parts_size
        else:
            end = len(boundaries)
        chunks_list.append(boundaries[start : end + 1])

    processed_docs = []
    with Pool(num_processes) as p:
        processed_docs = p.map(
            functools.partial(
                proccess_chunk,
                input_path=input_path,
                special_tokens=special_tokens,
                process_func=process_func,
            ),
            chunks_list,
        )
        processed_docs = reduce(operator.add, processed_docs)
        return processed_docs


def merge_pair(token_pair):
    return token_pair[0] + token_pair[1]


def merge_in_byte_word(new_token_pair, byte_word):
    if all(x in byte_word for x in new_token_pair):
        replace_indices = []
        for i in range(len(byte_word) - len(new_token_pair) + 1):
            if new_token_pair == byte_word[i : i + len(new_token_pair)]:
                replace_indices.append(i)
        for replace_idx in reversed(replace_indices):
            byte_word = (
                byte_word[:replace_idx]
                + tuple([merge_pair(new_token_pair)])
                + byte_word[replace_idx + len(new_token_pair) :]
            )
    return byte_word


def get_merge_indices(new_token_pair, byte_word_pairs):
    if new_token_pair in byte_word_pairs:
        replace_indices = []
        for i, byte_word_pair in enumerate(byte_word_pairs):
            if new_token_pair == byte_word_pair:
                replace_indices.append(i)
        return replace_indices


def merge_in_indices(new_token_pair, byte_word, replace_indices):
    for replace_idx in reversed(replace_indices):
        byte_word = (
            byte_word[:replace_idx]
            + tuple([merge_pair(new_token_pair)])
            + byte_word[replace_idx + len(new_token_pair) :]
        )
    return byte_word
