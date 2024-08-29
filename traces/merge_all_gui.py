      
import glob
import gzip
import json
import os
import pathlib
import sys
from typing import Optional

# install:
#    tkinterdnd2-universal : refer to https://github.com/pmgagne/tkinterdnd2/issues/7

def unzip(infile, outfile):
    with gzip.open(infile, "rb") as f:
        with open(outfile, "wb") as fout:
            fout.write(f.read())


def unzip_if_not_exist(infile: str, outfile: Optional[str] = None):
    if outfile is None:
        assert infile.endswith(".gz") or infile.endswith(".tgz")
        outfile = pathlib.Path(infile).with_suffix("")
        print(f"outfile is not set, using input without .gz suffix: {outfile}")
    if pathlib.Path(outfile).is_file:
        print(f"{outfile} already exist from pathlib, skip")
        if os.path.exists(outfile):
            print(f"{outfile} already exist, skip")
            return
    unzip(infile, outfile)


def unzip_jsons(indir):
    print(f"unzip all gz files from: {indir}")
    to_merge_files = glob.glob(f"{indir}/*json.gz") + glob.glob(f"{indir}/*json.tgz")
    for infile in to_merge_files:
        unzip_if_not_exist(infile)


def merge_json(indir, output_json):
    print(f"merge json from {indir} to {output_json}")
    unzip_jsons(indir)
    events = []
    to_merge_files = glob.glob(f"{indir}/*json")

    for tl_file in to_merge_files:
        if tl_file.endswith("merged.json"):
            continue
        with open(tl_file, "r") as f:
            full_tl_json = json.load(f)

        rank = full_tl_json["distributedInfo"]["rank"]
        world_size = full_tl_json["distributedInfo"]["world_size"]
        for e in full_tl_json["traceEvents"]:
            e["pid"] = f"{e['pid']}_{rank}"
            if isinstance(e["tid"], int):
                e["tid"] = e['tid'] * world_size + rank
            if e["name"] == "thread_name":
                e["args"]["name"] = f'{e["args"]["name"]}_{rank}'
            if e["name"] == "thread_sort_index":
                e["args"]["sort_index"] = e["args"]["sort_index"] * world_size + rank
        events.extend(full_tl_json["traceEvents"])

    with open(output_json, "w") as f:
        full_tl_json["traceEvents"] = events
        json.dump(events, f)


def merge_json_callback(e):
    callback_data = e.data
    print(f"callback_data: {type(callback_data)} `{callback_data}`")

    for indir in callback_data.split():
        indir = pathlib.Path(indir)
        if not indir.is_dir():
            print(f"{indir} is not a valid directory")
            return
        merged_jsons = list(indir.glob("*merged.json"))
        if merged_jsons:
            print(f"{merged_jsons} already exists")
            return
        output_json = indir / "merged.json"
        merge_json(indir, output_json)
        lb.insert(tk.END, f"merge {indir.name} done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default=None)
    args = parser.parse_args()

    if args.indir is not None:
        merge_json(args.indir, pathlib.Path(args.indir) / "merged.json")
        sys.exit(0)


    import tkinter as tk

    from tkinterdnd2 import DND_FILES, TkinterDnD
    root = TkinterDnD.Tk()  # notice - use this instead of tk.Tk()
    lb = tk.Listbox(root)
    lb.insert(1, "drag files to here")

    # register the listbox as a drop target
    lb.drop_target_register(DND_FILES)
    lb.dnd_bind("<<Drop>>", merge_json_callback)

    lb.pack()
    root.mainloop()

    