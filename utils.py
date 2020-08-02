import subprocess
import os
import logging

def check_remove_dir(path, force = False):
    """
    Checks if path exists.
    If so, asks the user to either remove it. If user does not remove it, quit.
    If not, create path and out_img_path folders.
    """
    while os.path.isdir(path):
        if not force:
            delete = input(f"{path} could not be created, folder already exists. Do you want to remove this folder? (y/n) \n")
        if force or delete == "y":
            try:
                rm_tmp = subprocess.run(["rm", "-r", path], check=True,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
                logging.info(f"{path} has been removed.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.stderr))
        elif delete == "n":
            logging.error("Cannot pursue without deleting folder.")
            sys.exit()
        else:
            logging.error("Please enter a valid answer (y or n).")

def bbox_area(bot, top, right, left):
    #TODO: move to utils
    return (bot - top) * (right - left)
