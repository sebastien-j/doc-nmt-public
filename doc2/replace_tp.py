import argparse

def main(fname, gname):
    with open(fname) as f:
        with open(gname, 'w') as g:
            for line in f:
                line = line.replace('|@@ |@@ |', '|||')
                g.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('gname')

    args = parser.parse_args()

    main(args.fname, args.gname)
