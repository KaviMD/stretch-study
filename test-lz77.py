# https://www.programmersought.com/article/55755735846/
# https://timguite.github.io/jekyll/update/2020/03/15/lz77-in-python.html
# https://rosettacode.org/wiki/LZW_compression#Python

with open('data/simplified.txt', 'r') as f:
    string = f.read()
'''
buffer = ""
search_buf_length, look_ahead_buf_length, str_length = 25, 6, len(string)
search_buf_pos, look_ahead_buf_pos = 0, search_buf_length
encode_list = []

def Init():
    global buffer
    buffer = string[search_buf_pos:search_buf_pos+search_buf_length]
    for i in buffer:
        encode_list.append([0,0,i])
    buffer += string[look_ahead_buf_pos:look_ahead_buf_pos+look_ahead_buf_length]

def MoveForward(step):
    global search_buf_pos, look_ahead_buf_pos, buffer
    search_buf_pos += step; look_ahead_buf_pos += step
    buffer = string[search_buf_pos:search_buf_pos+search_buf_length+look_ahead_buf_length]

def Encode():
    sym_offset = search_buf_length
    max_length, max_offset, next_sym = 0, 0, buffer[sym_offset]
    buffer_length = len(buffer)
    if buffer_length - sym_offset == 1:
        encode_list.append([0,0,next_sym])
        return max_length
    for offset in range(1,search_buf_length+1):
        pos = sym_offset - offset
        n = 0
        while buffer[pos + n] == buffer[sym_offset + n]:
            n += 1
            if n == buffer_length - search_buf_length - 1: break
        if max_length < n:
            max_length = n
            max_offset = offset
            next_sym = buffer[sym_offset+n]
    encode_list.append([max_offset, max_length, next_sym])
    return max_length

def LZ77():
    while 1:
        step = Encode() + 1
        MoveForward(step)
        if look_ahead_buf_pos >= str_length: break

Init()
LZ77()
for i in encode_list:
    print(i)

def Decode(encode_lise):
    ans = ''
    for i in encode_list:
        offset, length, sym = i
        for j in range(length):
            ans += ans[-offset]
        ans += sym
    return ans

print(Decode(encode_list))
'''
################################################################################################

import logging


def compress(
    input_string: str, max_offset: int = 2047, max_length: int = 31
) -> [(int, int, str)]:
    """Compress the input string into a list of length, offset, char values"""

    # Create the input
    input_array = str(input_string[:])

    # Create a string of the characters which have been passed
    window = ""

    ## Store output in this list
    output = []

    while input_array != "":
        length, offset = best_length_offset(window, input_array, max_length, max_offset)
        output.append((offset, length, input_array[0]))
        window += input_array[:length]
        input_array = input_array[length:]

    return output


def best_length_offset(
    window: str, input_string: str, max_length: int = 15, max_offset: int = 4095
) -> (int, int):
    """Take the window and an input string and return the offset and length
    with the biggest length of the input string as a substring"""

    if max_offset < len(window):
        cut_window = window[-max_offset:]
    else:
        cut_window = window

    # Return (0, 0) if the string provided is empty
    if input_string is None or input_string == "":
        return (0, 0)

    # Initialise result parameters - best case so far
    length, offset = (1, 0)

    # This should also catch the empty window case
    if input_string[0] not in cut_window:
        best_length = repeating_length_from_start(input_string[0], input_string[1:])
        return (min((length + best_length), max_length), offset)

    # Best length now zero to allow occurences to take priority
    length = 0

    # Test for every string in the window, in reverse order to keep the offset as low as possible
    # Look for either the whole window or up to max offset away, whichever is smaller
    for index in range(1, (len(cut_window) + 1)):
        # Get the character at this offset
        char = cut_window[-index]
        if char == input_string[0]:
            found_offset = index
            # Collect any further strings which can be found
            found_length = repeating_length_from_start(
                cut_window[-index:], input_string
            )
            if found_length > length:
                length = found_length
                offset = found_offset

    # Only return up to the maximum length
    # This will capture the maximum number of characters allowed
    # although it might not capture the maximum amount of characters *possible*
    return (min(length, max_length), offset)


def repeating_length_from_start(window: str, input_string: str) -> int:
    """Get the maximum repeating length of the input from the start of the window"""
    if window == "" or input_string == "":
        return 0

    if window[0] == input_string[0]:
        return 1 + repeating_length_from_start(
            window[1:] + input_string[0], input_string[1:]
        )
    else:
        return 0


def compress_file(input_file: str, output_file: str):
    """Open and read an input file, compress it, and write the compressed
    values to the output file"""
    try:
        with open(input_file) as f:
            input_array = f.read()
    except FileNotFoundError:
        print(f"Could not find input file at: {input_file}")
        raise
    except Exception:
        raise

    compressed_input = to_bytes(compress(input_array))

    with open(output_file, "wb") as f:
        f.write(compressed_input)

print(compress(string, len(string)))