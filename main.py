"""
main.py  --  THE REMOTE CONTROL. Run one command, it calls the right step.

    python main.py build            step 1+2+3: read files, chunk, embed, save index
    python main.py chunks           show every chunk (see what the system "knows")
    python main.py ask "question"   steps 4+5: retrieve + answer
    python main.py chat             ask many questions interactively
    python main.py eval             step 6: test retrieval quality

Optional filters for ask/chat:  --doc Garden   --type "table content"
"""
import argparse

import config
from step3_index import build_index
from step4_retrieve import Retriever
from step5_answer import answer


def show_chunks():
    from step2_chunk import build_all_chunks
    for c in build_all_chunks():
        print(f"\n{c['id']}  [{c['type']} / {c['sub_type']}]  section: {c['location']}\n{c['text']}")


def ask(retriever, question, args):
    hits = retriever.search(question, args.k, doc=args.doc, type_=args.type)
    print("\nRetrieved chunks:")
    for n, (score, c) in enumerate(hits, 1):
        print(f"  [{n}] {score:.2f}  {c['doc']} | {c['location']} | {c['sub_type']}")
    reply = answer(question, hits)
    if reply:
        print(f"\nAnswer:\n{reply}\n")
    else:
        print("\n(Retrieval-only mode: set ANTHROPIC_API_KEY and `pip install anthropic` for generated answers.)")
        print(f"Top chunk:\n{hits[0][1]['text']}\n" if hits else "No matches.\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "chunks", "ask", "chat", "eval"])
    p.add_argument("question", nargs="?")
    p.add_argument("--k", type=int, default=config.TOP_K, help="chunks to retrieve")
    p.add_argument("--doc", help="only search documents whose name contains this")
    p.add_argument("--type", help="only search this taxonomy TYPE, e.g. 'table content'")
    p.add_argument("--backend", default=config.EMBED_BACKEND, help="auto | sbert | tfidf")
    args = p.parse_args()

    if args.command == "build":
        build_index(args.backend)
    elif args.command == "chunks":
        show_chunks()
    elif args.command == "eval":
        from step6_evaluate import run
        run(args.k)
    elif args.command == "ask":
        if not args.question:
            p.error('give a question: python main.py ask "How many panels at Site B?"')
        ask(Retriever(), args.question, args)
    elif args.command == "chat":
        r = Retriever()
        while (q := input("\nQuestion (blank to quit): ").strip()):
            ask(r, q, args)


if __name__ == "__main__":
    main()
