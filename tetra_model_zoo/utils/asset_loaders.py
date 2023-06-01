import os
import sys
import threading
import time
from typing import Any, Callable, Optional

import gdown
import requests
from git import Repo
from PIL import Image

MODEL_ZOO_STORE = os.path.expanduser("~/.tetra/model-zoo")
MODEL_ZOO_ASSET_PATH = (
    "https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo"
)


def _query_yes_no(question, default="yes"):
    """
    Ask a yes/no question and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Sourced from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt, end="")
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def _get_model_dir(model_name: str) -> str:
    model_dir = os.path.join(MODEL_ZOO_STORE, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def maybe_clone_git_repo(git_file_path: str, commit_hash, model_name: str) -> str:
    """Clone (or pull) a repository, save it to disk in a standard location,
    and return the absolute path to the cloned location."""

    # http://blah.come/author/name.git -> name, author
    repo_name = os.path.basename(git_file_path).split(".")[0]
    repo_author = os.path.basename(os.path.dirname(git_file_path))
    local_path = os.path.join(
        _get_model_dir(model_name), f"{repo_author}_{repo_name}_git"
    )

    if not os.path.exists(os.path.join(local_path, ".git")):
        # Clone repo
        should_clone = _query_yes_no(
            f"{model_name} requires repository {git_file_path} . Ok to clone?",
        )
        if should_clone:
            print(f"Cloning {git_file_path}to {local_path}...")
            repo = Repo.clone_from(git_file_path, local_path)
            repo.git.checkout(commit_hash)
            print("Done")
        else:
            raise ValueError(
                f"Unable to load {model_name} without its required repository."
            )

    return local_path


class SourceAsRoot:
    THREAD_LOCK = threading.Lock()

    """
    Context manager that runs code with:
     * the source repository added to the system path,
     * cwd set to the source repo's root directory.

    Only one of this class should be active per Python session.
    """

    def __init__(
        self,
        source_repo_url: str,
        source_repo_commit_hash: str,
        source_repo_name: str,
    ):
        self.source_repo_url = source_repo_url
        self.source_repo_commit_hash = source_repo_commit_hash
        self.source_repo_name = source_repo_name

    def __enter__(self):
        SourceAsRoot.THREAD_LOCK.acquire()
        self.repository_path = maybe_clone_git_repo(
            self.source_repo_url, self.source_repo_commit_hash, self.source_repo_name
        )
        self.cwd = os.getcwd()

        # Patch path for this load only, since the model source
        # code references modules via a global scope.
        sys.path.append(self.repository_path)
        os.chdir(self.repository_path)

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Reset global state
        os.chdir(self.cwd)
        sys.path.remove(self.repository_path)
        SourceAsRoot.THREAD_LOCK.release()


def maybe_download_s3_data(s3_path: str, model_name: str) -> str:
    """
    Parameters:
    - s3_path: path relative to tetra-public-assets/model-zoo/
    - model_name: download artifacts to MODEL_ZOO_STORE/<model_name>

    Returns:
    - The local filepath of the download data.
    """
    url = f"{MODEL_ZOO_ASSET_PATH}/{s3_path}"
    return download_data(url, model_name)


def download_data(url: str, model_name: str):
    """
    Downloads data from the internet and stores it in the same directory as
    other assets for the model. Returns the local filepath of the download data.
    """
    dst_path = os.path.join(_get_model_dir(model_name), url.rsplit("/", 1)[-1])
    if not os.path.exists(dst_path):
        print(f"Downloading data at {url} to {dst_path}... ", end="")
        file_data = requests.get(url)
        if file_data.status_code != 200:
            raise ValueError(f"Unable to download file at {url}")
        with open(dst_path, "wb") as dst_file:
            dst_file.write(file_data.content)
        print("Done")
    return dst_path


def download_google_drive(file_id: str, model_name: str, filename: str):
    """
    Download file from google drive to the local directory.

    Parameters:
        file_id: Unique identifier of the file in Google Drive.
            Typically found in the URL.
        model_name: Model for which this asset is being downloaded.
            Used to choose where in the local filesystem to put it.
        filename: Filename under which it will be saved locally.

    Returns:
        Filepath within the local filesystem.
    """
    dst_path = os.path.join(_get_model_dir(model_name), filename)
    if not os.path.exists(dst_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading data at {url} to {dst_path}... ", end="")
        gdown.download(url, dst_path, quiet=False)
        print("Done")
    return dst_path


def load_image(image_path: str, model_name: str) -> Image.Image:
    """Loads an image from the specified path.
    Will first download the image to the appropriate standard location if image_path is a URL."""
    if image_path.startswith("http"):
        image_path = download_data(image_path, model_name)

    return Image.open(image_path)


def callback_with_retry(
    num_retries: int,
    callback: Callable,
    *args: Optional[Any],
    **kwargs: Optional[Any],
) -> Any:
    """Allow retries when running provided function."""
    if num_retries == 0:
        raise RuntimeError(f"Unable to run function {callback.__name__}")
    else:
        try:
            return callback(*args, **kwargs)
        except Exception as error:
            error_msg = (
                f"Error: {error.message}"
                if hasattr(error, "message")
                else f"Error: {str(error)}"
            )
            print(error_msg)
            if hasattr(error, "status_code"):
                print(f"Status code: {error.status_code}")
            time.sleep(10)
            return callback_with_retry(num_retries - 1, callback, *args, **kwargs)
