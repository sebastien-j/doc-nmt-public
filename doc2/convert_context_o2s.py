import argparse

def main(prefix, suffix, left, right, output):
    # TODO: Exception handling

    infiles = []
    for ii in xrange(left, 0, -1):
        infiles.append(open(prefix+'l'+str(ii)+suffix))
    for ii in xrange(1, right+1):
        infiles.append(open(prefix+'r'+str(ii)+suffix))

    with open(output, 'w') as f:
        for line in infiles[0]:
            cur_lines = []
            cur_lines.append(line[:-1]) # Remove \n
            for ii in xrange(1, len(infiles)):
                l_ = infiles[ii].readline()[:-1]
                if l_ != '':
                    cur_lines.append(l_)
            f.write(' ||| '.join(cur_lines) + '\n')

    for g in infiles:
        g.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix')
    parser.add_argument('--suffix')
    parser.add_argument('--left', type=int, default=5)
    parser.add_argument('--right', type=int, default=5)
    parser.add_argument('--output')

    args = parser.parse_args()

    main(args.prefix, args.suffix, args.left, args.right, args.output)

