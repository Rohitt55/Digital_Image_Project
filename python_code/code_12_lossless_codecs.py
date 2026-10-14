"""Small lossless teaching codecs with independently decodable bitstreams."""
import bisect
from collections import Counter
import heapq
import struct


def pack_bits(bits):
    """Store eight bits in each byte, starting with the highest bit."""
    bits = list(bits)
    output = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        byte_index = index // 8
        bit_position = 7 - index % 8
        output[byte_index] |= int(bit) << bit_position
    return (bytes(output), len(bits))


def unpack_bits(data, count):
    return [(data[index // 8] >> (7 - index % 8)) & 1 for index in range(count)]


def rle_encode(values):
    """Replace each repeated run with a (count, value) pair."""
    pairs = []
    for value in values:
        if pairs and pairs[-1][1] == value:
            pairs[-1] = (pairs[-1][0] + 1, value)
        else:
            pairs.append((1, value))
    return (b''.join((struct.pack('>IB', count, value) for count, value in pairs)), {'runs': len(pairs)})


def rle_decode(payload):
    result = []
    for index in range(0, len(payload), 5):
        count, value = struct.unpack('>IB', payload[index:index + 5])
        result.extend([value] * count)
    return result


def huffman_encode(values):
    """Merge the two least frequent nodes until one coding tree remains."""
    counts = Counter(values)
    heap = [(count, index, value) for index, (value, count) in enumerate(sorted(counts.items()))]
    heapq.heapify(heap)
    serial = len(heap)
    while len(heap) > 1:
        left_count, _, left_node = heapq.heappop(heap)
        right_count, _, right_node = heapq.heappop(heap)
        parent = (left_node, right_node)
        heapq.heappush(heap, (left_count + right_count, serial, parent))
        serial += 1
    table = {}

    def visit(node, prefix):
        if isinstance(node, int):
            table[node] = prefix or '0'
        else:
            visit(node[0], prefix + '0')
            visit(node[1], prefix + '1')
    visit(heap[0][2], '')
    payload, count = pack_bits((int(bit) for value in values for bit in table[value]))
    codes = {str(symbol): code for symbol, code in table.items()}
    return payload, {'bit_count': count, 'codes': codes}


def huffman_decode(payload, metadata):
    reverse = {value: int(key) for key, value in metadata['codes'].items()}
    prefix = ''
    result = []
    for bit in unpack_bits(payload, metadata['bit_count']):
        prefix += str(bit)
        if prefix in reverse:
            result.append(reverse[prefix])
            prefix = ''
    if prefix:
        raise ValueError('Incomplete Huffman stream.')
    return result


def arithmetic_encode(values):
    """Narrow a 32-bit interval for each symbol and emit its stable bits."""
    counts = Counter(values)
    symbols = sorted(counts)
    frequencies = [counts[symbol] for symbol in symbols]
    cumulative = [0]
    for frequency in frequencies:
        cumulative.append(cumulative[-1] + frequency)
    indexes = {symbol: index for index, symbol in enumerate(symbols)}
    total = len(values)
    low = 0
    high = (1 << 32) - 1
    pending = 0
    bits = []
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 * quarter

    def emit(bit):
        nonlocal pending
        bits.append(bit)
        bits.extend([1 - bit] * pending)
        pending = 0
    for value in values:
        # Select this symbol's subinterval using cumulative frequencies.
        index = indexes[value]
        width = high - low + 1
        high = low + width * cumulative[index + 1] // total - 1
        low = low + width * cumulative[index] // total
        while True:
            # Emit a stable leading bit, or defer bits near the midpoint.
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low *= 2
            high = high * 2 + 1
    pending += 1
    emit(0 if low < quarter else 1)
    payload, count = pack_bits(bits)
    return (payload, {'symbols': symbols, 'frequencies': frequencies, 'symbol_count': total, 'bit_count': count})


def arithmetic_decode(payload, metadata):
    """Reverse interval updates using the frequencies stored in metadata."""
    symbols = metadata['symbols']
    cumulative = [0]
    for frequency in metadata['frequencies']:
        cumulative.append(cumulative[-1] + frequency)
    total = cumulative[-1]
    bits = unpack_bits(payload, metadata['bit_count'])
    position = 0

    def read():
        nonlocal position
        bit = bits[position] if position < len(bits) else 0
        position += 1
        return bit
    code = 0
    for _ in range(32):
        code = code * 2 + read()
    low = 0
    high = (1 << 32) - 1
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 * quarter
    result = []
    for _ in range(metadata['symbol_count']):
        width = high - low + 1
        scaled = ((code - low + 1) * total - 1) // width
        index = bisect.bisect_right(cumulative, scaled) - 1
        result.append(symbols[index])
        high = low + width * cumulative[index + 1] // total - 1
        low = low + width * cumulative[index] // total
        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                code -= half
            elif low >= quarter and high < three_quarters:
                low -= quarter
                high -= quarter
                code -= quarter
            else:
                break
            low *= 2
            high = high * 2 + 1
            code = code * 2 + read()
    return result


def golomb_encode(values, method, m=16, k=4):
    """Encode a quotient followed by its remainder; Rice uses divisor 2**k."""
    bits = []
    for value in values:
        if method == 'Exp-Golomb':
            quotient = (value >> k) + 1
            binary = format(quotient, 'b')
            bits.extend([0] * (len(binary) - 1))
            bits.extend(map(int, binary))
            if k:
                bits.extend(map(int, format(value % 2 ** k, f'0{k}b')))
        else:
            divisor = 2 ** k if method == 'Rice' else m
            quotient, remainder = divmod(value, divisor)
            bits.extend([1] * quotient)
            bits.append(0)
            width = (divisor - 1).bit_length()
            cutoff = 2 ** width - divisor
            if width:
                if remainder < cutoff:
                    binary = format(remainder, f'0{width - 1}b')
                else:
                    binary = format(remainder + cutoff, f'0{width}b')
                bits.extend(map(int, binary))
    payload, count = pack_bits(bits)
    return (payload, {'method': method, 'm': m, 'k': k, 'bit_count': count, 'symbol_count': len(values)})


def golomb_decode(payload, metadata):
    bits = unpack_bits(payload, metadata['bit_count'])
    position = 0
    result = []

    def read(count=1):
        nonlocal position
        if position + count > len(bits):
            raise ValueError('Incomplete Golomb stream.')
        value = 0
        for bit in bits[position:position + count]:
            value = value * 2 + bit
        position += count
        return value
    method = metadata['method']
    k = metadata['k']
    for _ in range(metadata['symbol_count']):
        if method == 'Exp-Golomb':
            zeros = 0
            while read() == 0:
                zeros += 1
            quotient = (1 << zeros) + read(zeros)
            result.append(((quotient - 1) << k) + read(k))
        else:
            divisor = 2 ** k if method == 'Rice' else metadata['m']
            quotient = 0
            while read() == 1:
                quotient += 1
            width = (divisor - 1).bit_length()
            cutoff = 2 ** width - divisor
            remainder = read(width - 1) if width else 0
            if width and remainder >= cutoff:
                remainder = remainder * 2 + read() - cutoff
            result.append(quotient * divisor + remainder)
    return result


def lzw_encode(values):
    """Build a dictionary of byte sequences and output 16-bit codes."""
    dictionary = {bytes([index]): index for index in range(256)}
    next_code = 256
    current = b''
    codes = []
    for value in values:
        symbol = bytes([value])
        combined = current + symbol
        if combined in dictionary:
            current = combined
        else:
            codes.append(dictionary[current])
            if next_code < 65536:
                dictionary[combined] = next_code
                next_code += 1
            current = symbol
    if current:
        codes.append(dictionary[current])
    return (b''.join((struct.pack('>H', code) for code in codes)), {
        'code_count': len(codes),
        'dictionary_size': next_code,
        'code_width_bits': 16
    })


def lzw_decode(payload):
    codes = [struct.unpack('>H', payload[index:index + 2])[0] for index in range(0, len(payload), 2)]
    if not codes:
        return []
    dictionary = {index: bytes([index]) for index in range(256)}
    current = dictionary[codes[0]]
    output = bytearray(current)
    next_code = 256
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = current + current[:1]
        else:
            raise ValueError('Invalid LZW code.')
        output.extend(entry)
        if next_code < 65536:
            dictionary[next_code] = current + entry[:1]
            next_code += 1
        current = entry
    return list(output)
