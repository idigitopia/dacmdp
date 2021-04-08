import argparse

def print_args(args, to_show_args=[], title="Local"):
    if not to_show_args:
        title = "All Arguments"

    header = "#" * 50 + " " * 4 + title + " " * 4 + "#" * 50
    print(header)
    iter_args = iter(list(vars(args)))  # small hack to iter 2 from a list at a time
    for arg1 in iter_args:
        if to_show_args and arg1 not in to_show_args:
            continue
        try:
            arg2 = next(iter_args)
            while to_show_args and arg2 not in to_show_args:
                arg2 = next(iter_args)
            print(arg1.ljust(30), ":", str(getattr(args, arg1)).ljust(30), arg2.ljust(30), ":", getattr(args, arg2))
        except:
            print(arg1.ljust(30), ":", str(getattr(args, arg1)).ljust(30))
    print("#" * len(header))