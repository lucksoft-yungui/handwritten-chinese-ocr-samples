from unittest.mock import patch
from main import main
from io import StringIO
import sys


def test_train():
    testargs = ["main",
                "-m", "hctr",
                "-d", "/Users/peiyandong/Documents/code/ai/hw_train_data",
                "-dl", "./train-data-label/chineseocr",
                "-dlf", "rec_digit_label",
                "-b", "4",
                "-pf", "100",
                "-lr", "1e-4"
                ]
    with patch('sys.stdout', new=StringIO()) as fake_out:
        with patch.object(sys, 'argv', testargs):
            main()
