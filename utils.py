import subprocess
import os
import logging
import urllib
import zipfile
import requests
import cv2
import shutil
import sys

def check_remove_dir(path, force=False, mobile_shell=False, remove_required=True):
    """
    Checks if path exists.
    If so, asks the user to either remove it. If user does not remove it, quit.
    If not, create path and out_img_path folders.
    Args:
        mobile_shell: bool
            if mobile_shell, we assume that the folder "path" already exists. It needs to be removed. Otherwise, keep asking until folder is removed or user says "n".
    """
    while os.path.isdir(path) or mobile_shell:
        if not force:
            if remove_required:
                delete = input(f"Folder {path} already exists. Are you sure you want to remove this folder? (y/n) \n")
            else:
                delete = input(f"Do you want to remove the folder located at {path}? [y/n] \n")
        if force or delete == "y":
            try:
                rm_cmd = ["rm", "-r", path]
                if mobile_shell:
                    rm_cmd = ["adb", "shell"] + rm_cmd
                rm_tmp = subprocess.run(rm_cmd, check=True,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
                logging.info(f"{path} has been removed.")
                break
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.stderr))
        elif delete == "n":
            if remove_required:
                logging.error("Cannot pursue without deleting folder.")
            sys.exit()
        else:
            logging.error("Please enter a valid answer (y or n).")


def download_required_files(url,
                           folder_path,
                           file_name,
                           unzip_folder=None,
                           force=False):


    req = urllib.request.Request(url, method="HEAD")
    size_file = urllib.request.urlopen(req).headers["Content-Length"]


    download = "n"
    while download != "y":
        if not force:
            download = input(f"You are about to download {file_name} ({size_file} bytes) to the folder {folder_path}. Do you want to continue? [y/n] \n")
        if force or download == "y":
            logging.info(f"Downloading {file_name} ({size_file} bytes) from {url} to folder {folder_path}...")
            zip_path, hdrs = urllib.request.urlretrieve(url, os.path.join(folder_path, file_name))
            logging.info(f"Extracting {zip_path} to folder {folder_path}...")
            with zipfile.ZipFile(f"{zip_path}", 'r') as z:
                z.extractall(f"{folder_path}")
            #self.input_data_path = zip_path[:-4]
            break
        elif download == "n":
            logging.error(f"Cannot pursue without downloading {file_name}.")
            sys.exit()
        else:
            logging.error("Please enter a valid answer (y or n).")



def process_single_img(img_path, new_img_path, img_size):
    """
    Processes a single image.
    If img_size is specified, rescales the image.
    Otherwise, just copies to the new path.
    Args:
        img_path: str
            path to the image to process
        new_img_path: str
            output path to the new image
    """
    if img_size is None:
        logging.debug(f"Copying {img_path} to \n {new_img_path}")
        shutil.copyfile(img_path,
                        new_img_path)
    else:
        #logging.debug(f"Rescaling {img_path} to shape {self.new_img_size} and save to \n {new_img_path}")
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(new_img_path, resized_img)
