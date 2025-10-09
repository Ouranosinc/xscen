"""A collection of various convenience objects and functions to use in scripts."""

import logging
import mimetypes
import os
import shutil as sh
import signal
import smtplib
import sys
import time
import warnings
from contextlib import contextmanager
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from traceback import format_exception

import xarray as xr
from matplotlib.figure import Figure

from .catalog import ProjectCatalog
from .config import parse_config
from .utils import get_cat_attrs


logger = logging.getLogger(__name__)

__all__ = [
    "TimeoutException",
    "measure_time",
    "move_and_delete",
    "save_and_update",
    "send_mail",
    "send_mail_on_exit",
    "skippable",
    "timeout",
]


@parse_config
def send_mail(
    *,
    subject: str,
    msg: str,
    to: str | None = None,
    server: str = "127.0.0.1",
    port: int = 25,
    attachments: None | (list[tuple[str, Figure | os.PathLike] | Figure | os.PathLike]) = None,
) -> None:
    """
    Send email.

    Email a single address through a login-less SMTP server.
    The default values of server and port should work out-of-the-box on Ouranos's systems.

    Parameters
    ----------
    subject: str
      Subject line.
    msg: str
      Main content of the email. Can be UTF-8 and multi-line.
    to: str, optional
      Email address to which send the email. If None (default), the email is sent to "{os.getlogin()}@{os.uname().nodename}".
      On unix systems simply put your real email address in `$HOME/.forward` to receive the emails sent to this local address.
    server : str
      SMTP server url. Defaults to 127.0.0.1, the local host. This function does not try to log-in.
    port: int
      Port of the SMTP service on the server. Defaults to 25, which is usually the default port on unix-like systems.
    attachments : list of paths or matplotlib figures or tuples of a string and a path or figure, optional
      List of files to attach to the email.
      Elements of the list can be paths, the mimetypes of those is guessed and the files are read and sent.
      Elements can also be matplotlib Figures which are send as png image (savefig) with names like "Figure00.png".
      Finally, elements can be tuples of a filename to use in the email and the attachment, handled as above.

    Returns
    -------
    None
    """
    # Inspired by https://betterprogramming.pub/how-to-send-emails-with-attachments-using-python-dd37c4b6a7fd
    email = EmailMessage()
    email["Subject"] = subject
    email["From"] = f"{os.getlogin()}@{os.uname().nodename}"
    email["To"] = to or f"{os.getlogin()}@{os.uname().nodename}"
    email.set_content(msg)

    for i, att in enumerate(attachments or []):
        fname = None
        if isinstance(att, tuple):
            fname, att = att
        if isinstance(att, Figure):
            data = BytesIO()
            att.savefig(data, format="png")
            data.seek(0)
            email.add_attachment(
                data.read(),
                maintype="image",
                subtype="png",
                filename=fname or f"Figure{i:02d}.png",
            )
        else:  # a path
            attpath = Path(att)
            ctype, encoding = mimetypes.guess_type(attpath)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)

            with attpath.open("rb") as fp:
                email.add_attachment(
                    fp.read(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=fname or attpath.name,
                )

    with smtplib.SMTP(host=server, port=port) as SMTP:
        SMTP.send_message(email)


class ExitWatcher:
    """An object that watches system exits and exceptions before the python exits."""

    def __init__(self):
        self.code = None
        self.error = None
        self.hooked = False

    def hook(self):
        """Hooks the watcher to the system by monkeypatching `sys` with its own methods."""
        if not self.hooked:
            self.orig_exit = sys.exit
            self.orig_excepthook = sys.excepthook
            sys.exit = self.exit
            sys.excepthook = self.err_handler
            self.hooked = True
        else:
            warnings.warn("Exit hooks have already been overridden.", stacklevel=2)

    def unhook(self):
        if self.hooked:
            sys.exit = self.orig_exit
            sys.excepthook = self.orig_excepthook
        else:
            raise ValueError("Exit hooks were not overridden. Cannot unhook.")

    def exit(self, code=0):
        self.code = code
        self.orig_exit(code)

    def err_handler(self, *exc_info):
        self.error = exc_info
        self.orig_excepthook(*exc_info)


exit_watcher = ExitWatcher()
exit_watcher.hook()


@parse_config
def send_mail_on_exit(
    *,
    subject: str | None = None,
    msg_ok: str | None = None,
    msg_err: str | None = None,
    on_error_only: bool = False,
    skip_ctrlc: bool = True,
    **mail_kwargs,
) -> None:
    """
    Send an email with content depending on how the system exited.

    This function is best used by registering it with `atexit`. Calls :py:func:`send_mail`.

    Parameters
    ----------
    subject : str, optional
        Email subject. Will be appended by "Success", "No errors" or "Failure" depending
        on how the system exits.
    msg_ok : str, optional
        Content of the email if the system exists successfully.
    msg_err : str, optional
        Content of the email id the system exists with a non-zero code or with an error.
        The message will be appended by the exit code or with the error traceback.
    on_error_only : boolean
        Whether to only send an email on a non-zero/error exit.
    skip_ctrlc : boolean
        If True (default), exiting with a KeyboardInterrupt will not send an email.
    mail_kwargs
        Other arguments passed to :py:func:`send_mail`.
        The `to` argument is necessary for this function to work.

    Example
    -------
    Send an eamil titled "Woups" upon non-successful program exit. We assume the `to`
    field was given in the config.

    >>> import atexit
    >>> atexit.register(send_mail_on_exit, subject="Woups", on_error_only=True)
    """
    subject = subject or "Workflow"
    msg_err = msg_err or "Workflow exited with some errors."
    if not on_error_only and exit_watcher.error is None and exit_watcher.code in [None, 0]:
        send_mail(
            subject=subject + " - Success",
            msg=msg_ok or "Workflow exited successfully.",
            **mail_kwargs,
        )
    elif exit_watcher.error is None and (exit_watcher.code or 0) > 0:
        send_mail(
            subject=subject + " - No errors",
            msg=f"{msg_err}\nSystem exited with code {exit_watcher.code}.",
            **mail_kwargs,
        )
    elif exit_watcher.error is not None and (exit_watcher.error[0] is not KeyboardInterrupt or not skip_ctrlc):
        tb = "".join(format_exception(*exit_watcher.error))
        msg_err = f"{msg_err}\n\n{tb}"
        send_mail(subject=subject + " - Failure", msg=msg_err, **mail_kwargs)


@parse_config
class measure_time:
    """
    Context for timing a code block.

    Parameters
    ----------
    name : str, optional
      A name to give to the block being timed, for meaningful logging.
    cpu : boolean
      If True, the CPU time is also measured and logged.
    logger : logging.Logger, optional
      The logger object to use when sending Info messages with the measured time.
      Defaults to a logger from this module.
    """

    def __init__(
        self,
        name: str | None = None,
        cpu: bool = False,
        logger: logging.Logger = logger,
    ):
        self.name = name or ""
        self.cpu = cpu
        self.logger = logger

    def __enter__(self):  # noqa: D105
        self.start = time.perf_counter()
        self.start_cpu = time.process_time()
        msg = f"Started process {self.name}."
        self.logger.info(msg)
        return

    def __exit__(self, *args, **kwargs):  # noqa: D105
        elapsed = time.perf_counter() - self.start
        elapsed_cpu = time.process_time() - self.start_cpu
        occ = elapsed_cpu / elapsed
        s = f"Process {self.name} done in {elapsed:.02f} s"
        if self.cpu:
            s += f" and used {elapsed_cpu:.02f} of cpu time ({occ:.1%} % occupancy)."

        self.logger.info(s)


# FIXME: This should be written as "TimeoutError"
class TimeoutException(Exception):  # noqa: N818
    """An exception raised with a timeout occurs."""

    def __init__(self, seconds: int, task: str = "", **kwargs):
        self.msg = f"Task {task} timed out after {seconds} seconds"
        super().__init__(self.msg, **kwargs)


@contextmanager
def timeout(seconds: int, task: str = ""):
    """
    Timeout context manager.

    Only one can be used at a time, this is not multithread-safe : it cannot be used in
    another thread than the main one, but multithreading can be used in parallel.

    Parameters
    ----------
    seconds : int
      Number of seconds after which the context exits with a TimeoutException.
      If None or negative, no timeout is set and this context does nothing.
    task : str, optional
      A name to give to the task, allowing a more meaningful exception.
    """
    if seconds is None or seconds <= 0:
        yield
    else:
        # FIXME: These variables are not used
        def _timeout_handler(signum, frame):  # noqa: F841
            raise TimeoutException(seconds, task)

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def skippable(seconds: int = 2, task: str = "", logger: logging.Logger | None = None):
    """
    Skippable context manager.

    When CTRL-C (SIGINT, KeyboardInterrupt) is sent within the context,
    this catches it, prints to the log and gives a timeout during which a subsequent
    interruption will stop the script. Otherwise, the context exits normally.

    This is meant to be used within a loop so that we can skip some iterations:

    .. code-block:: python

        for i in iterable:
            with skippable(2, i):
                some_skippable_code()

    Parameters
    ----------
    seconds: int
      Number of seconds to wait for a second CTRL-C.
    task : str
      A name for the skippable task, to have an explicit script.
    logger : logging.Logger, optional
      The logger to use when printing the messages. The interruption signal is
      notified with ERROR, while the skipping is notified with INFO.
      If not given (default), a brutal print is used.
    """
    if logger:
        err = logger.error
        inf = logger.info
    else:
        err = inf = print
    try:
        yield
    except KeyboardInterrupt:
        err("Received SIGINT. Do it again to stop the script.")
        time.sleep(seconds)
        inf(f"Skipping {task}.")


def save_and_update(
    ds: xr.Dataset,
    pcat: ProjectCatalog,
    path: str | os.PathLike | None = None,
    file_format: str | None = None,
    build_path_kwargs: dict | None = None,
    save_kwargs: dict | None = None,
    update_kwargs: dict | None = None,
):
    """
    Construct the path, save and delete.

    This function can be used after each task of a workflow.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to save.
    pcat: ProjectCatalog
        Catalog to update after saving the dataset.
    path: str or os.pathlike, optional
        Path where to save the dataset.
        If the string contains variables in curly bracket. They will be filled by catalog attributes.
        If None, the `catutils.build_path` function will be used to create a path.
    file_format: {'nc', 'zarr'}
        Format of the file.
        If None, look for the following in order: build_path_kwargs['format'], a suffix in path, ds.attrs['cat:format'].
        If nothing is found, it will default to zarr.
    build_path_kwargs: dict, optional
        Arguments to pass to `build_path`.
    save_kwargs: dict, optional
        Arguments to pass to `save_to_netcdf` or `save_to_zarr`.
    update_kwargs: dict, optional
        Arguments to pass to `update_from_ds`.
    """
    build_path_kwargs = build_path_kwargs or {}
    save_kwargs = save_kwargs or {}
    update_kwargs = update_kwargs or {}

    # try to guess file format if not given.
    if file_format is None:
        if "format" in build_path_kwargs:
            file_format = build_path_kwargs.get("format")
        elif path is not None and Path(path).suffix:
            file_format = Path(path).suffix.split(".")[-1]
        else:
            file_format = ds.attrs.get("cat:format", "zarr")

    # get path
    if path is not None:
        path = str(path).format(**get_cat_attrs(ds, var_as_str=True))  # fill path with attrs
    else:  # if path is not given build it
        build_path_kwargs.setdefault("format", file_format)
        from .catutils import build_path

        path = build_path(ds, **build_path_kwargs)

    # save
    if file_format == "zarr":
        from .io import save_to_zarr

        save_to_zarr(ds, path, **save_kwargs)
    elif file_format == "nc":
        from .io import save_to_netcdf

        save_to_netcdf(ds, path, **save_kwargs)
    else:
        raise ValueError(f"file_format {file_format} is not valid. Use zarr or nc.")

    # update catalog
    pcat.update_from_ds(ds=ds, path=path, **update_kwargs)

    msg = f"File {path} has been saved successfully and the catalog was updated."
    logger.info(msg)


def move_and_delete(
    moving: list[list[str | os.PathLike]],
    pcat: ProjectCatalog,
    deleting: list[str | os.PathLike] | None = None,
    copy: bool = False,
):
    """
    First, move files, then update the catalog with new locations. Finally, delete directories.

    This function can be used at the end of for loop in a workflow to clean temporary files.

    Parameters
    ----------
    moving : list of lists of str or os.PathLike
        list of lists of path of files to move, following the format: [[source 1, destination1], [source 2, destination2],...]
    pcat : ProjectCatalog
        Catalog to update with new destinations
    deleting : list of str or os.PathLike, optional
        List of directories to be deleted, including all contents, and recreated empty. e.g. The working directory of a workflow.
    copy : bool, optional
        If True, copy directories instead of moving them.
    """
    if isinstance(moving, list) and isinstance(moving[0], list):
        for files in moving:
            source, dest = files[0], files[1]
            if Path(source).exists():
                if copy:
                    msg = f"Copying {source} to {dest}."
                    logger.info(msg)
                    copied_files = sh.copytree(source, dest, dirs_exist_ok=True)
                    for f in copied_files:
                        # copied files don't include zarr files
                        if f[-16:] == ".zarr/.zmetadata":
                            with warnings.catch_warnings():
                                # Silence RuntimeWarning about failed guess of backend engines
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                ds = xr.open_dataset(f[:-11])
                            pcat.update_from_ds(ds=ds, path=f[:-11])
                        if f[-3:] == ".nc":
                            ds = xr.open_dataset(f)
                            pcat.update_from_ds(ds=ds, path=f)
                else:
                    msg = f"Moving {source} to {dest}."
                    logger.info(msg)
                    sh.move(source, dest)
                if Path(dest).suffix in [".zarr", ".nc"]:
                    with warnings.catch_warnings():
                        # Silence RuntimeWarning about failed guess of backend engines
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        ds = xr.open_dataset(dest)
                    pcat.update_from_ds(ds=ds, path=dest)
            else:
                msg = f"You are trying to move {source}, but it does not exist."
                logger.info(msg)
    else:
        raise ValueError("`moving` should be a list of lists.")

    # erase workdir content if this is the last step
    if isinstance(deleting, list):
        for dir_to_delete in deleting:
            if Path(dir_to_delete).exists() and Path(dir_to_delete).is_dir():
                msg = f"Deleting content inside {dir_to_delete}."
                logger.info(msg)
                sh.rmtree(dir_to_delete)
                Path(dir_to_delete).mkdir()
    elif deleting is None:
        pass
    else:
        raise ValueError("`deleting` should be a list.")
