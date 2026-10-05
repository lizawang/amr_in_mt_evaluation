from transition_amr_parser.parse import AMRParser
import os
import argparse



argp = argparse.ArgumentParser()
argp.add_argument("-i", "--input_path", type=str, help="The input .txt file path to parse.")
argp.add_argument("--lang", type=str, required=True, default="he-en", help="The language pair of the translation result.")
argp.add_argument("-m", "--model", type=str, default="AMR3-joint-ontowiki-seed43", help="Parser model name from IBM.")
argp.add_argument("-b", "--in_batch", action="store_true", help="Run in batches. Default false, aka, parse one sentence each time.")
argp.add_argument("--beam", type=int, default=10)

#langs = ["he-en", "zh-en"]

def main():
    # parse arguments
    args = argp.parse_args()
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    
    lang = args.lang
    print(f"Lang: ~~{lang}~~")
    result_fn = os.path.join(file_dir, "ibm_large")
    if "joint" in args.model:
        result_fn = os.path.join(file_dir, "ibm_ensemble")
    os.makedirs(result_fn, exist_ok=True)
    
    # Download and save a model named AMR3.0 to cache, with beam=10
    parser = AMRParser.from_pretrained(args.model, beam=args.beam, nbest=1, num_samples=None, sampling_topp=-1, temperature=1.0)
        
    # get all files of one language pair

    # for file in os.listdir(args.input_dir):
        # if not file.endswith(".txt"):
        #     continue
    annotations = []
    sents = []
    file_name = os.path.split(args.input_path)[-1]
    print(f"File: ~~~Parsing file: {file_name}~~~")
    sents_toks = []
    with open(f"{args.input_path}", "r", encoding="utf-8") as f:
        for sent in f.readlines():
            #print(f"sent:{sent}")
            #NOTE give a synthetic sent when sent is empty
            sent = sent.strip()
            if not sent:
                sent = "."
            tokens, positions = parser.tokenize(sent)
            sents_toks.append(tokens)
            sents.append(sent)  
            if not args.in_batch: 
                annotation, machine = parser.parse_sentence(tokens)
                annotations.append(annotation)
    # parse all sents in a file as a batch using parse_sentences
    print(f"~~~~~~~~~~total sents: {len(sents_toks)}~~~~~~~~~~~")
    if args.in_batch:
        annotations, machines = parser.parse_sentences(sents_toks, jamr=True, 
                                                       no_isi=True, 
                                                       batch_size=128,roberta_batch_size=128, 
                                                       beam=args.beam)
    
    with open(f"{result_fn}/{lang}.{file_name.split('.')[-2]}.amr", "w", encoding="utf-8") as f2:
        for a in annotations:
            f2.write("".join(a))
            f2.write("\n")
    print(f"~~~~write into {result_fn}/{lang}.{file_name.split('.')[-2]}.amr")
        
    
if __name__ == "__main__":
    main()