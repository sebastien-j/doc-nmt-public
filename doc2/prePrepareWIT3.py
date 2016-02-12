# Preprocess dev/test data before prepareWIT3.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
            "Preprocess WIT3 training data")
    parser.add_argument("input", help="Input file")
    return parser.parse_args()

def main():

    args = parse_args()
    
    restricted = ['<?xml', '<mteval', '<srcset', '<refset', '<doc', '<url', '<description', '<keywords', '<talkid', '<title', '</refset', '</srcset', '</mteval',
    '<h1', '</h1', '<p', '</p']

    with open(args.input) as f:
        with open(args.input[:-3]+'out.xml', 'w') as g:
            skip = False
            for line in f:
                for word in restricted:
                    if line.startswith(word):
                        skip = True
                        break
                if not skip:
                    if line.startswith('</doc>'):
                        g.write('<reviewer></reviewer>\n')
                    else:
                        try:
                            assert line.startswith('<seg id=') and line.endswith(' </seg>\n')
                        except:
                            pos = line.find(">")
                            line = line[:pos+1] + " " + line[pos+1:]
                            line = line.replace('</seg>\n', ' </seg>\n')
                            assert line.startswith('<seg id=') and line.endswith(' </seg>\n')
                        line = line.split()
                        line = ' '.join(line[2:-1]) + '\n'
                        g.write(line)
                skip = False

if __name__ == "__main__":
    main()
