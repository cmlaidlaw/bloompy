import math, ctypes, struct, hashlib, random, unittest, itertools

class BloomFilter(object):

    def __init__(self, capacity=1024, error=0.005):
        """
        Instantiates a BloomFilter

        Keyword arguments:
        capacity -- the maximum number of items that will be stored in the filter (default 1024)
        error -- the acceptable probability of false-positive results (default 0.005)
        """
        if capacity < 1:
            raise ValueError('Argument `capacity` must be greater than or equal to 1')
        if error < 0 or error > 1:
            raise ValueError('Argument `error` must be between 0 and 1.')

        self._capacity = capacity
        self._item_count = 0
        self._error = error
        self._filter_size = self._calculate_optimal_filter_size(self._capacity, self._error)
        self._hash_count = self._calculate_optimal_hash_count(self._filter_size, self._capacity)
        self._hash_functions = self._generate_hash_functions(self._hash_count)
        self._buffer_format_string = '<B'
        self._buffer = self._create_buffer(self._filter_size)


    # These methods constitute the filter's public interface

    def __contains__(self, item):
        """
        Checks for an item's membership in the Bloom Filter
        """
        hashes = self._hash_item(item)
        for hashed_value in hashes:
            if not self._get_bit(int(hashed_value, 16) % self._filter_size):
                return False
        return True

    def __len__(self):
        """
        Returns the number of items the Bloom Filter can support before exceeding the false-positive threshold.
        """
        return self._capacity

    def add(self, item):
        """
        Adds an item to the Bloom Filter
        """
        if self._item_count == self._capacity:
            raise StandardError('Exceeding the specified capacity will result in an unacceptable false-positive rate.')
        keys = self._hash_item(item)
        for key in keys:
            self._set_bit(int(key, 16) % self._filter_size)
        self._item_count += 1

    def query(self, item):
        """
        An alias of __contains__
        """
        return self.__contains__(item)


    # These methods support filter operations
    # The formulas for optimal filter size and hash count are taken from: http://en.wikipedia.org/wiki/Bloom_filter

    def _calculate_optimal_filter_size(self, capacity, error):
        return int(math.ceil(-((capacity * math.log(error)) / ((math.log(2))**2))))

    def _calculate_optimal_hash_count(self, size, capacity):
        return int(math.ceil((size / capacity) * math.log(2)))

    def _generate_hash_functions(self, count):
        return [self._generate_hash_function_with_prefix(index) for index in range(count)]

    def _generate_hash_function_with_prefix(self, prefix):
        # This is tremendously inefficient since we probably don't need cryptographic integrity.
        return lambda x: hashlib.sha256(str(prefix) + str(x)).hexdigest()

    def _hash_item(self, item):
        return [self._hash_functions[i](item) for i in range(self._hash_count)]


    # These methods deal with the raw binary representation of the filter

    def _create_buffer(self, size):
        if size < 1:
            size = 1 # Minimum buffer size of 1 byte
        num_bytes = 1 + int(math.floor(size / 8))
        return ctypes.create_string_buffer(num_bytes)

    def _calculate_internal_offset(self, offset):
        return (int(math.floor(offset / 8)), offset % 8)

    def _set_bit(self, index):
        byte_offset, bit_offset = self._calculate_internal_offset(index)
        byte_value = struct.unpack_from(self._buffer_format_string, self._buffer, byte_offset)[0]
        byte_value = byte_value | (1 << bit_offset)
        struct.pack_into(self._buffer_format_string, self._buffer, byte_offset, byte_value)

    def _get_bit(self, index):
        byte_offset, bit_offset = self._calculate_internal_offset(index)
        byte_value = struct.unpack_from(self._buffer_format_string, self._buffer, byte_offset)[0]
        return byte_value & (1 << bit_offset)


# Testing functions
def has_size(filter, size):
    return filter._filter_size == size

def bit_is_set(filter, index):
    return filter._get_bit(index)

# Tests
class BloomFilterTests(unittest.TestCase):

    def test_01_filter_buffer_is_correct_size(self):
        filter = BloomFilter(capacity=32, error=0.001)
        optimal_filter_size = int(math.ceil(-((32 * math.log(0.001)) / ((math.log(2))**2))))
        self.failIf(not has_size(filter, optimal_filter_size))

    def test_02_initialized_bit_not_set(self):
        filter = BloomFilter()
        self.failIf(bit_is_set(filter, 0))

    def test_03_set_bit_set(self):
        filter = BloomFilter()
        filter._set_bit(0)
        self.failIf(not bit_is_set(filter, 0))

    def test_04_random_set_bit_set(self):
        filter = BloomFilter()
        index = random.randint(0, filter._filter_size)
        filter._set_bit(index)
        self.failIf(not bit_is_set(filter, index))

    def test_05_each_hash_result_unique(self):
        filter = BloomFilter()
        keys = filter._hash_item('test')
        self.failIf(len(keys) > len(set(keys)))

    def test_06_public_add_method_sets_bits(self):
        filter = BloomFilter()
        keys = filter._hash_item('test')
        filter.add('test')
        for key in keys:
            self.failIf(not bit_is_set(filter, int(key, 16) % filter._filter_size))

    def test_07_contains_tests_membership(self):
        filter = BloomFilter()
        filter.add('test')
        self.failIf('test' not in filter)

    def test_08_public_query_method_aliases_contains(self):
        filter = BloomFilter()
        filter.add('test')
        self.failIf(not filter.query('test'))

    def test_09_public_query_method_collision_remains_unlikely(self):
        filter = BloomFilter()
        filter.add('test')
        self.failIf('Test' in filter)

    def test_10_exceeding_capacity_raises_error(self):
        filter = BloomFilter(capacity=1024)
        for x in range(1024):
            filter.add(x)
        self.failUnlessRaises(StandardError, filter.add, 1024)

    def test_11_normal_operation_stays_within_maximum_error(self):
        inserted = 0
        false_positives = 0
        filter = BloomFilter(capacity=65536, error=0.005)
        items = {str(random.randint(0,262144)):False for x in range(65536)}
        for item in itertools.islice(items.iterkeys(), len(items)/2):
            filter.add(item)
            items[item] = True
            inserted += 1
        for item, item_was_inserted in items.iteritems():
            if item in filter and not item_was_inserted:
                false_positives += 1
        self.failIf(false_positives/len(items) > 0.005)

def main():
    unittest.main()

if __name__ == '__main__':
    main()