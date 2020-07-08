import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
from pyjarowinkler import distance  # pip install pyjarowinkler

callnote_url = '/Users/zhinyuan.peng/Documents/DataBaseExport/Call2_vod__c.csv'

def note_clean(sent):
    """
    clean the input text
    :param sent: text
    :return: text
    """
    sent = re.sub('[,.:;?^$!+_()*\-\n\t/]', ' ', sent)
    sent_list = re.split("\'| |’", sent)
    sent_processed = []
    for word in sent_list:
        if word != "" and word != " ":
            sent_processed.append(word.strip(" ").lower())
    return " ".join(sent_processed)


def main():
    pass


if __name__ == "__main__":
    main()
