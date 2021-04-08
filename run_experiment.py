from exp_track.exp_track_helper import *
import exp_track.experiments as eXP

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--exp_ids", help="Name of the project", type=str,nargs="+", default=["none"])
parser.add_argument("--append", help="string to append at the end of the command", default="")
parser.add_argument("--remove", help="string to remove from the command", nargs="+", default=[""])
parser.add_argument("--replace", help="string to remove from the command", nargs=2, default=["", ""])
parser.add_argument("--replace1", help="string to remove from the command", nargs=2, default=["", ""])
parser.add_argument("--replace2", help="string to remove from the command", nargs=2, default=["", ""])
parser.add_argument("--replace3", help="string to remove from the command", nargs=2, default=["", ""])

args = parser.parse_args()
assert args.exp_ids[0] != "none", "Script name must be passed, at least one"

open("/tmp/{}.pk".format(args.exp_ids[0]),"wb")
print("Appending:{} Removing:{}".format(args.append, args.remove))

for exp_id in args.exp_ids:
    original_command = eXP.ExpPool.get_by_id(exp_id).command
    modified_command = original_command + " " + args.append
    for to_remove_string in args.remove:
        modified_command = modified_command.replace(to_remove_string, "")

    for replace_option in [args.replace, args.replace1, args.replace2, args.replace3]:
        to_replace, replace_with = replace_option
        print("replacing {} with {}".format(to_replace,replace_with))
        modified_command = modified_command.replace(to_replace,replace_with)

    modified_command =  ' '.join(modified_command.split())

    print("Running Experiment Id: ", exp_id)
    print("Original Command: ", original_command)
    print("Running Command: ", modified_command)
    os.system(modified_command)
    print("Execution complete moving to next command.")
