import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
            "Preprocess WIT3 training data")
    parser.add_argument("--source",
            default="/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.en.tok", help="Source file")
    parser.add_argument("--target",
            default="/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.fr.tok", help="Target file")
    parser.add_argument("--source-text",
            default=None, help="Source file, no document info")
    parser.add_argument("--target-text",
            default=None, help="Target file, no document info")
    parser.add_argument("--source-context",
            default=None, help="BOW source context")
    parser.add_argument("--target-context",
            default=None, help="BOW target context")
    parser.add_argument("--num-context-left", type=int, default=0, help="Number of sentences on the left")
    parser.add_argument("--num-context-right", type=int, default=0, help="Number of sentences on the right")
    parser.add_argument("--use-middle", action="store_true", default=False, help="Use current sentence as part of context")
    parser.add_argument("--split-sentences", action="store_true", default=False)
    parser.add_argument("--corpus", default="WIT3", help="Choice of 'WIT3' or 'europarl'")
    return parser.parse_args()

def main():

    """
    &lt; url &gt; http : / / www.ted.com / talks / david _ gallo _ on _ life _ in _ the _ deep _ oceans &lt; / url &gt;
    &lt; keywords &gt; talks , TED Conference , animals , geology , life , oceans , science , submarine , technology &lt; / keywords &gt;
    &lt; speaker &gt; David Gallo &lt; / speaker &gt;
    &lt; talkid &gt; 343 &lt; / talkid &gt;
    &lt; title &gt; Life in the deep oceans &lt; / title &gt;
    &lt; description &gt; With vibrant video clips captured by submarines , David Gallo takes us to some of Earth &apos;s darkest , most violent , toxic and beautiful habitats , the valleys and volcanic ridges of the oceans &apos; depths , where life is bizarre , resilient and shockingly abundant . &lt; / description &gt;
    This is Bill Lange . I &apos;m Dave Gallo .
    And we &apos;re going to tell you some stories from the sea here in video .
    We &apos;ve got some of the most incredible video of Titanic that &apos;s ever been seen , and we &apos;re not going to show you any of it .
    ...
    &lt; reviewer &gt; &lt; / reviewer &gt;
    &lt; translator &gt; &lt; / translator &gt;
    """
 
    args = parse_args()
    if args.source_text == None:
        args.source_text = args.source + '.text'
    if args.source_context == None:
        args.source_context = args.source + '.context'
    if args.target_text == None:
        args.target_text = args.target + '.text'
    if args.target_context == None:
        args.target_context = args.target + '.context'
    src = []
    trg = []
    src_doc = []
    trg_doc = []

    if args.corpus == 'WIT3':
        print "WIT3"
        with open(args.source) as f:
            with open(args.target) as g:
                for src_line in f:
                    trg_line = g.readline()
                    if not src_line.startswith("&lt;"):
                        src_doc.append(src_line.split())
                        trg_doc.append(trg_line.split())
                    elif src_line.startswith("&lt; reviewer"): # At the end of every document, including the last one
                        src.append(src_doc)
                        trg.append(trg_doc)
                        src_doc = []
                        trg_doc = []
    elif args.corpus == 'europarl':
        print "europarl"
        with open(args.source) as f:
            with open(args.target) as g:
                for src_line in f:
                    trg_line = g.readline()
                    if not src_line.startswith("&lt;"):
                        src_doc.append(src_line.split())
                        trg_doc.append(trg_line.split())
                    elif src_line.startswith("&lt; P"):
                        pass
                    elif src_line.startswith("&lt; SPEAKER") or src_line.startswith("&lt; CHAPTER"): # Before every chapter, or change of speaker
                        if src_doc != []:
                            assert trg_doc != []
                            src.append(src_doc)
                            trg.append(trg_doc)
                            src_doc = []
                            trg_doc = []
                if src_doc != []:
                    assert trg_doc != []
                    src.append(src_doc)
                    trg.append(trg_doc)

    num_docs = len(src)
    assert num_docs == len(trg)
    
    src_exists = os.path.isfile(args.source_text)
    trg_exists = os.path.isfile(args.target_text)

    if not src_exists and not trg_exists:
        print "Writing source and target text files"
        with open(args.source_text, 'w') as f:
            with open(args.target_text, 'w') as g:
                for ii in xrange(num_docs):
                    doc_len = len(src[ii])
                    assert doc_len == len(trg[ii])
                    for jj in xrange(doc_len):
                        f.write(" ".join(src[ii][jj]) + '\n')
                        g.write(" ".join(trg[ii][jj]) + '\n')
    elif not src_exists and trg_exists:
        raise Exception("The source text file does not exist, but the target does")
    elif src_exists and not trg_exists:
        raise Exception("The target text file does not exist, but the source does")
    else:
        print "Source and target text files exist"

    src_context_exists = os.path.isfile(args.source_context)
    trg_context_exists = os.path.isfile(args.target_context)

    if not src_context_exists and not trg_context_exists:
        print "Writing source and target context files"
        with open(args.source_context, 'w') as context_f:
            with open(args.target_context, 'w') as context_g:
                ext = []
                if args.split_sentences:
                    ext = ['|||']
                for ii in xrange(num_docs):
                    doc_len = len(src[ii])
                    assert doc_len == len(trg[ii])
                    for jj in xrange(doc_len):
                        src_tmp = []
                        trg_tmp = []
                        for kk in xrange(max(0, jj - args.num_context_left), jj):
                            src_tmp.extend(src[ii][kk] + ext)
                            trg_tmp.extend(trg[ii][kk] + ext)
                        if args.use_middle:
                            src_tmp.extend(src[ii][jj] + ext)
                            trg_tmp.extend(trg[ii][jj] + ext)
                        for kk in xrange(jj+1, min(doc_len, jj + args.num_context_right + 1)):
                            src_tmp.extend(src[ii][kk] + ext)
                            trg_tmp.extend(trg[ii][kk] + ext)
                        if args.split_sentences: # Remove last extension
                            src_tmp = src_tmp[:-1]
                            trg_tmp = trg_tmp[:-1]

                        context_f.write(" ".join(src_tmp) + '\n')
                        context_g.write(" ".join(trg_tmp) + '\n')
    elif not src_context_exists and trg_context_exists:
        raise Exception("The source context file does not exist, but the target does")
    elif src_context_exists and not trg_context_exists:
        raise Exception("The target context file does not exist, but the source does")
    else:
        print "Source and target context files exist"

if __name__ == "__main__":
    main()
