# This file store all the functions dealing with text


def write_txt_ann(out_path, txt, ann):
    """
    write the txt and corresponding annotation to out_path
    This\t"O"
    is\t"O"
    an\t"O"
    example\t"O"
    \n
    This\t"O"
    is\t"O"
    the\t"O"
    next\t"O"
    one\t"O"
    Args:
        out_path: out path of the file
        txt: a list of words
        ann: a list of tags
    Returns:
        no return
    """
    with open(out_path, "a+") as f:
        for word, tag in zip(txt, ann):
            s = word + "\t" + tag + "\n"
            f.write(s)
        f.write("\n")


def main():
    pass


if __name__ == "__main__":
    main()

