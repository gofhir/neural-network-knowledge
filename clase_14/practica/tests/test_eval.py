import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _eval import build_char_maps

def test_build_char_maps_shakespeare():
    text = open(os.path.join(os.path.dirname(__file__), "..", "shakespeare.txt")).read()
    c2i, i2c = build_char_maps(text)
    assert len(c2i) == 65
    assert all(c2i[i2c[i]] == i for i in i2c)
