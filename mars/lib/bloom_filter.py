# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# pylint: disable=superfluous-parens,redefined-variable-type
# superfluous-parens: Sometimes extra parens are more clear

"""Bloom Filter: Probabilistic set membership testing for large sets"""

# Shamelessly borrowed (under MIT license) from
# https://code.activestate.com/recipes/577686-bloom-filter/
# About Bloom Filters: https://en.wikipedia.org/wiki/Bloom_filter

# Tweaked by Daniel Richard Stromberg, mostly to:
# 1) Give it a little nicer __init__ parameters.
# 2) Improve the hash functions to get a much lower rate of false positives.
# 3) Give it a selection of backends.
# 4) Make it pass pylint.

# In the literature:
# k is the number of probes - we call this num_probes_k
# m is the number of bits in the filter - we call this num_bits_m
# n is the ideal number of elements to eventually be stored in the filter - we
# call this ideal_num_elements_n
# p is the desired error rate when full - we call this error_rate_p

import array
import math
import os
import random

try:
    import mmap as mmap_mod
except ImportError:
    # Jython lacks mmap()
    HAVE_MMAP = False
else:
    HAVE_MMAP = True


class Mmap_backend(object):
    """
    Backend storage for our "array of bits" using an mmap'd file.
    Please note that this has only been tested on Linux so far.
    """

    effs = 2**8 - 1

    def __init__(self, num_bits, filename):
        if not HAVE_MMAP:
            raise NotImplementedError("mmap is not available")
        self.num_bits = num_bits
        self.num_chars = (self.num_bits + 7) // 8
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_BINARY"):
            flags |= getattr(os, "O_BINARY")
        self.file_ = os.open(filename, flags)
        os.lseek(self.file_, self.num_chars + 1, os.SEEK_SET)
        os.write(self.file_, b"\x00")
        self.mmap = mmap_mod.mmap(self.file_, self.num_chars)

    def is_set(self, bitno):
        """Return true iff bit number bitno is set"""
        byteno, bit_within_wordno = divmod(bitno, 8)
        mask = 1 << bit_within_wordno
        byte = self.mmap[byteno]
        return byte & mask

    def set(self, bitno):
        """set bit number bitno to true"""

        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        byte = self.mmap[byteno]
        byte |= mask
        self.mmap[byteno] = byte

    def clear(self, bitno):
        """clear bit number bitno - set it to false"""

        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        byte = self.mmap[byteno]
        byte &= Mmap_backend.effs - mask
        self.mmap[byteno] = byte

    def __iand__(self, other):
        assert self.num_bits == other.num_bits

        for byteno in range(self.num_chars):
            self.mmap[byteno] = self.mmap[byteno] & other.mmap[byteno]

        return self

    def __ior__(self, other):
        assert self.num_bits == other.num_bits

        for byteno in range(self.num_chars):
            self.mmap[byteno] = self.mmap[byteno] | other.mmap[byteno]

        return self

    def close(self):
        """Close the file"""
        os.close(self.file_)


class File_seek_backend(object):
    """Backend storage for our "array of bits" using a file in which we seek"""

    effs = 2**8 - 1

    def __init__(self, num_bits, filename):
        self.num_bits = num_bits
        self.num_chars = (self.num_bits + 7) // 8
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_BINARY"):
            flags |= getattr(os, "O_BINARY")
        self.file_ = os.open(filename, flags)
        os.lseek(self.file_, self.num_chars + 1, os.SEEK_SET)
        os.write(self.file_, b"\x00")

    def is_set(self, bitno):
        """Return true iff bit number bitno is set"""
        byteno, bit_within_wordno = divmod(bitno, 8)
        mask = 1 << bit_within_wordno
        os.lseek(self.file_, byteno, os.SEEK_SET)
        byte = os.read(self.file_, 1)[0]
        return byte & mask

    def set(self, bitno):
        """set bit number bitno to true"""

        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        os.lseek(self.file_, byteno, os.SEEK_SET)
        byte = os.read(self.file_, 1)[0]
        byte |= mask
        os.lseek(self.file_, byteno, os.SEEK_SET)
        os.write(self.file_, bytes([byte]))

    def clear(self, bitno):
        """clear bit number bitno - set it to false"""

        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        os.lseek(self.file_, byteno, os.SEEK_SET)
        byte = os.read(self.file_, 1)[0]
        byte &= File_seek_backend.effs - mask
        os.lseek(self.file_, byteno, os.SEEK_SET)
        os.write(self.file_, bytes([byte]))

    # These are quite slow ways to do iand and ior, but they should work,
    # and a faster version is going to take more time
    def __iand__(self, other):
        assert self.num_bits == other.num_bits

        for bitno in range(self.num_bits):
            if self.is_set(bitno) and other.is_set(bitno):
                self.set(bitno)
            else:
                self.clear(bitno)

        return self

    def __ior__(self, other):
        assert self.num_bits == other.num_bits

        for bitno in range(self.num_bits):
            if self.is_set(bitno) or other.is_set(bitno):
                self.set(bitno)
            else:
                self.clear(bitno)

        return self

    def close(self):
        """Close the file"""
        os.close(self.file_)


class Array_then_file_seek_backend(object):
    # pylint: disable=R0902
    # R0902: We kinda need a bunch of instance attributes
    """
    Backend storage for our "array of bits" using a python array of integers up
    to some maximum number of bytes, then spilling over to a file.
    This is -not- a cache; we instead save the leftmost bits in RAM, and the
    rightmost bits (if necessary) in a file.  On open, we read from the file to
    RAM.  On close, we write from RAM to the file.
    """

    effs = 2**8 - 1

    def __init__(self, num_bits, filename, max_bytes_in_memory):
        self.num_bits = num_bits
        num_chars = (self.num_bits + 7) // 8
        self.filename = filename
        self.max_bytes_in_memory = max_bytes_in_memory
        self.bits_in_memory = min(num_bits, self.max_bytes_in_memory * 8)
        self.bits_in_file = max(self.num_bits - self.bits_in_memory, 0)
        self.bytes_in_memory = (self.bits_in_memory + 7) // 8
        self.bytes_in_file = (self.bits_in_file + 7) // 8

        self.array_ = array.array("B", [0]) * self.bytes_in_memory
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_BINARY"):
            flags |= getattr(os, "O_BINARY")
        self.file_ = os.open(filename, flags)
        os.lseek(self.file_, num_chars + 1, os.SEEK_SET)
        os.write(self.file_, b"\x00")

        os.lseek(self.file_, 0, os.SEEK_SET)
        offset = 0
        intended_block_len = 2**17
        while True:
            if offset + intended_block_len < self.bytes_in_memory:
                block = os.read(self.file_, intended_block_len)
            elif offset < self.bytes_in_memory:
                block = os.read(self.file_, self.bytes_in_memory - offset)
            else:
                break
            for index_in_block, byte in enumerate(block):
                self.array_[offset + index_in_block] = byte
            offset += intended_block_len

    def is_set(self, bitno):
        """Return true iff bit number bitno is set"""
        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        if byteno < self.bytes_in_memory:
            return self.array_[byteno] & mask
        else:
            os.lseek(self.file_, byteno, os.SEEK_SET)
            byte = os.read(self.file_, 1)[0]
            return byte & mask

    def set(self, bitno):
        """set bit number bitno to true"""
        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        if byteno < self.bytes_in_memory:
            self.array_[byteno] |= mask
        else:
            os.lseek(self.file_, byteno, os.SEEK_SET)
            byte = os.read(self.file_, 1)[0]
            byte |= mask
            os.lseek(self.file_, byteno, os.SEEK_SET)
            os.write(self.file_, bytes([byte]))

    def clear(self, bitno):
        """clear bit number bitno - set it to false"""
        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = Array_backend.effs - (1 << bit_within_byteno)
        if byteno < self.bytes_in_memory:
            self.array_[byteno] &= mask
        else:
            os.lseek(self.file_, byteno, os.SEEK_SET)
            byte = os.read(self.file_, 1)[0]
            byte &= File_seek_backend.effs - mask
            os.lseek(self.file_, byteno, os.SEEK_SET)
            os.write(self.file_, bytes([byte]))

    # These are quite slow ways to do iand and ior, but they should work,
    # and a faster version is going to take more time
    def __iand__(self, other):
        assert self.num_bits == other.num_bits

        for bitno in range(self.num_bits):
            if self.is_set(bitno) and other.is_set(bitno):
                self.set(bitno)
            else:
                self.clear(bitno)

        return self

    def __ior__(self, other):
        assert self.num_bits == other.num_bits

        for bitno in range(self.num_bits):
            if self.is_set(bitno) or other.is_set(bitno):
                self.set(bitno)
            else:
                self.clear(bitno)

        return self

    def close(self):
        """
        Write the in-memory portion to disk, leave the already-on-disk  portion
        unchanged
        """

        os.lseek(self.file_, 0, os.SEEK_SET)
        os.write(self.file_, bytes(self.array_[0 : self.bytes_in_memory]))

        os.close(self.file_)


class Array_backend(object):
    """
    Backend storage for our "array of bits" using a python array of integers
    """

    # Note that this has now been split out into a bits_mod for the benefit of
    # other projects.
    effs = 2**32 - 1

    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_words = (self.num_bits + 31) // 32
        self.array_ = array.array("L", [0]) * self.num_words

    def is_set(self, bitno):
        """Return true iff bit number bitno is set"""
        wordno, bit_within_wordno = divmod(bitno, 32)
        mask = 1 << bit_within_wordno
        return self.array_[wordno] & mask

    def set(self, bitno):
        """set bit number bitno to true"""
        wordno, bit_within_wordno = divmod(bitno, 32)
        mask = 1 << bit_within_wordno
        self.array_[wordno] |= mask

    def clear(self, bitno):
        """clear bit number bitno - set it to false"""
        wordno, bit_within_wordno = divmod(bitno, 32)
        mask = Array_backend.effs - (1 << bit_within_wordno)
        self.array_[wordno] &= mask

    # It'd be nice to do __iand__ and __ior__ in a base class, but
    # that'd be Much slower

    def __iand__(self, other):
        assert self.num_bits == other.num_bits

        for wordno in range(self.num_words):
            self.array_[wordno] &= other.array_[wordno]

        return self

    def __ior__(self, other):
        assert self.num_bits == other.num_bits

        for wordno in range(self.num_words):
            self.array_[wordno] |= other.array_[wordno]

        return self

    def close(self):
        """Noop for compatibility with the file+seek backend"""
        pass


def get_bitno_seed_rnd(bloom_filter, key):
    """
    Apply num_probes_k hash functions to key.
    Generate the array index and bitmask corresponding to each result.
    """

    # We're using key as a seed to a pseudorandom number generator
    hasher = random.Random(key).randrange
    for dummy in range(bloom_filter.num_probes_k):
        bitno = hasher(bloom_filter.num_bits_m)
        yield bitno % bloom_filter.num_bits_m


MERSENNES1 = [2**x - 1 for x in [17, 31, 127]]
MERSENNES2 = [2**x - 1 for x in [19, 67, 257]]


def simple_hash(int_list, prime1, prime2, prime3):
    """Compute a hash value from a list of integers and 3 primes"""
    result = 0
    for integer in int_list:
        result += ((result + integer + prime1) * prime2) % prime3
    return result


def hash1(int_list):
    """Basic hash function #1"""
    return simple_hash(int_list, MERSENNES1[0], MERSENNES1[1], MERSENNES1[2])


def hash2(int_list):
    """Basic hash function #2"""
    return simple_hash(int_list, MERSENNES2[0], MERSENNES2[1], MERSENNES2[2])


def get_filter_bitno_probes(bloom_filter, key):
    """
    Apply num_probes_k hash functions to key.
    Generate the array index and bitmask corresponding to each result
    """

    # This one assumes key is either bytes or str (or other list of integers)

    if hasattr(key, "__divmod__"):
        int_list = []
        temp = key
        while temp:
            quotient, remainder = divmod(temp, 256)
            int_list.append(remainder)
            temp = quotient
    elif isinstance(key, (list, tuple, str, bytes)) and not key:
        int_list = []

    elif isinstance(key, (list, tuple)):
        int_list = []
        for v in key:
            if isinstance(v, str):
                int_list.extend([ord(char) for char in v])
            elif hasattr(v, "__divmod__"):
                int_list.append(v)
            else:
                raise TypeError("Sorry, I do not know how to hash this type")
    elif isinstance(key[0], str):
        int_list = [ord(char) for char in key]
    else:
        raise TypeError("Sorry, I do not know how to hash this type")

    hash_value1 = hash1(int_list)
    hash_value2 = hash2(int_list)
    probe_value = hash_value1

    for _ in range(1, bloom_filter.num_probes_k + 1):
        probe_value *= hash_value1
        probe_value += hash_value2
        probe_value %= MERSENNES1[2]
        yield probe_value % bloom_filter.num_bits_m


def try_unlink(filename):
    """unlink a file.  Don't complain if it's not there"""
    try:
        os.unlink(filename)
    except OSError:
        pass
    return


class BloomFilter(object):
    """Probabilistic set membership testing for large sets"""

    def __init__(
        self,
        max_elements=10000,
        error_rate=0.1,
        probe_bitnoer=get_filter_bitno_probes,
        filename=None,
        start_fresh=False,
    ):
        # pylint: disable=R0913
        # R0913: We want a few arguments
        if max_elements <= 0:
            raise ValueError("ideal_num_elements_n must be > 0")
        if not (0 < error_rate < 1):
            raise ValueError("error_rate_p must be between 0 and 1 exclusive")

        self.error_rate_p = error_rate
        # With fewer elements, we should do very well. With more elements, our
        # error rate "guarantee" drops rapidly.
        self.ideal_num_elements_n = max_elements

        numerator = -1 * self.ideal_num_elements_n * math.log(self.error_rate_p)
        denominator = math.log(2) ** 2
        real_num_bits_m = numerator / denominator
        self.num_bits_m = int(math.ceil(real_num_bits_m))

        if filename is None:
            self.backend = Array_backend(self.num_bits_m)
        elif isinstance(filename, tuple) and isinstance(filename[1], int):
            if start_fresh:
                try_unlink(filename[0])
            if filename[1] == -1:
                self.backend = Mmap_backend(self.num_bits_m, filename[0])
            else:
                self.backend = Array_then_file_seek_backend(
                    self.num_bits_m,
                    filename[0],
                    filename[1],
                )
        else:
            if start_fresh:
                try_unlink(filename)
            self.backend = File_seek_backend(self.num_bits_m, filename)

        # AKA num_offsetters
        # Verified against
        # https://en.wikipedia.org/wiki/Bloom_filter#Probability_of_false_positives
        real_num_probes_k = (self.num_bits_m / self.ideal_num_elements_n) * math.log(2)
        self.num_probes_k = int(math.ceil(real_num_probes_k))
        self.probe_bitnoer = probe_bitnoer

    def __repr__(self):
        return (
            "BloomFilter(ideal_num_elements_n=%d, error_rate_p=%f, " + "num_bits_m=%d)"
        ) % (
            self.ideal_num_elements_n,
            self.error_rate_p,
            self.num_bits_m,
        )

    def add(self, key):
        """Add an element to the filter"""
        for bitno in self.probe_bitnoer(self, key):
            self.backend.set(bitno)

    def __iadd__(self, key):
        self.add(key)
        return self

    def _match_template(self, bloom_filter):
        """
        Compare a sort of signature for two bloom filters.
        Used in preparation for binary operations
        """
        return (
            self.num_bits_m == bloom_filter.num_bits_m
            and self.num_probes_k == bloom_filter.num_probes_k
            and self.probe_bitnoer == bloom_filter.probe_bitnoer
        )

    def union(self, bloom_filter):
        """Compute the set union of two bloom filters"""
        self.backend |= bloom_filter.backend

    def __ior__(self, bloom_filter):
        self.union(bloom_filter)
        return self

    def intersection(self, bloom_filter):
        """Compute the set intersection of two bloom filters"""
        self.backend &= bloom_filter.backend

    def __iand__(self, bloom_filter):
        self.intersection(bloom_filter)
        return self

    def __contains__(self, key):
        for bitno in self.probe_bitnoer(self, key):
            if not self.backend.is_set(bitno):
                return False
        return True

    def close(self):
        self.backend.close()
        self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.backend = None

    def __del__(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None
