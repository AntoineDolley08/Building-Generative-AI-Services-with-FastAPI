def test_chunk_text():
    text = "Testing GenAI services"
    results = chunk(text)
    assert len(results) == 2

def chunk(tokens: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

from unittest import result

import pytest
from rag.transform import chunk
def test_chunking_success():
    # GIVEN
    tokens = [1, 2, 3, 4, 5]
    # WHEN
    result = chunk(tokens, chunk_size=2)
    # THEN
    assert result == [[1, 2], [3, 4], [5]]
    
import pytest
from rag.transform import chunk
# GIVEN
@pytest.fixture(scope="module")
def tokens():
    return [1, 2, 3, 4, 5]

def test_token_chunking_small(token_list):
    result = chunk(tokens, chunk_size=2)
    assert result == [[1, 2], [3, 4], [5]]
    
def test_token_chunking_large(token_list):
    result = chunk(tokens, chunk_size=5)
    assert result == [[1, 2, 3, 4, 5]]

@pytest.mark.parametrize("tokens, chunk_size, expected", [
    ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]), # valid
    ([1, 2, 3, 4, 5], 3, [[1, 2, 3], [4, 5]]), # valid
    ([1, 2, 3, 4, 5], 1, [[1], [2], [3], [4], [5]]), # valid
    ([], 3, []), # valid/empty input
    418 | Chapter 11: Testing AI Services
    ([1, 2, 3], 5, [[1, 2, 3]]), # boundary input
    ([1, 2, 3, 4, 5], 0, "ValueError"), # invalid (chunk_size <= 0)
    ([1, 2, 3, 4, 5], -1, "ValueError"), # invalid (chunk_size <= 0)
    (
    list(range(10000)), 1000, [list(range(i, i + 1000)) # huge data
    for i in range(0, 10000, 1000)]
    )
    ])
def test_token_chunking(tokens, chunk_size, expected):
    if expected == "ValueError":
        with pytest.raises(ValueError):
            chunk(tokens, chunk_size)
    else:
        assert chunk(tokens, chunk_size) == expected