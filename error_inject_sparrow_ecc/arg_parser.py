import argparse

def parser_init():
    parser = argparse.ArgumentParser()

    parser.add_argument("--BER",
                        action="store",
                        dest="BER",
                        type=str,
                        help="enter BER")
    
    parser.add_argument("--zero_to_one",
                        action="store",
                        dest="zero_to_one",
                        type=str,
                        help="enter zero_to_one proportion")
    
    return parser