import unittest
from utils.data_process import note_clean


class MyTestCase(unittest.TestCase):

    def test_clean_lowcase(self):
        sent = "THIS IS A TEST to test the functioN Convert Upper case to LOWer CaSe"
        sent_clean = note_clean(sent)
        sent_clean_true = "this is a test to test the function convert upper case to lower case"
        self.assertEqual(sent_clean, sent_clean_true)

    def test_clean_blanks(self):
        sent = "   This is to test the   function to  remove head, tail blanks and duplicate blanks    "
        sent_clean = note_clean(sent)
        sent_clean_true = "this is to test the function to remove head tail blanks and duplicate blanks"
        self.assertEqual(sent_clean, sent_clean_true)

    def test_clean_special(self):
        sent = "This is to remove ?! and () and ' and $ and , and . and ; and ^ and _ and - + A/b"
        sent_clean = note_clean(sent)
        sent_clean_true = "this is to remove and and and and and and and and and a b"
        self.assertEqual(sent_clean, sent_clean_true)


if __name__ == '__main__':
    unittest.main()

