"""A collection of various convenience objects and functions to use in scripts."""
import logging
import mimetypes
import os
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
from typing import List, Optional, Tuple, Union

from matplotlib.figure import Figure

from .config import parse_config

logger = logging.getLogger(__name__)

__all__ = [
    "send_mail",
    "send_mail_on_exit",
    "measure_time",
    "timeout",
    "TimeoutException",
    "skippable",
]


@parse_config
def send_mail(
    *,
    subject: str,
    msg: str,
    to: str = None,
    server: str = "127.0.0.1",
    port: int = 25,
    attachments: Optional[
        List[Union[Tuple[str, Union[Figure, os.PathLike]], Figure, os.PathLike]]
    ] = None,
) -> None:
    """Send email.

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
            warnings.warn("Exit hooks have already been overrided.")

    def unhook(self):
        if self.hooked:
            sys.exit = self.orig_exit
            sys.excepthook = self.orig_excepthook
        else:
            raise ValueError("Exit hooks were not overriden, can't unhook.")

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
    subject: Optional[str] = None,
    msg_ok: Optional[str] = None,
    msg_err: Optional[str] = None,
    on_error_only: bool = False,
    skip_ctrlc: bool = True,
    **mail_kwargs,
) -> None:
    """Send an email with content depending on how the system exited.

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

    Returns
    -------
    None

    Example
    -------
    Send an eamil titled "Woups" upon non-successful program exit. We assume the `to`
    field was given in the config.

    >>> import atexit
    >>> atexit.register(send_mail_on_exit, subject='Woups', on_error_only=True)
    """
    subject = subject or "Workflow"
    msg_err = msg_err or "Workflow exited with some errors."
    if (
        not on_error_only
        and exit_watcher.error is None
        and exit_watcher.code in [None, 0]
    ):
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
    elif exit_watcher.error is not None and (
        exit_watcher.error[0] is not KeyboardInterrupt or not skip_ctrlc
    ):
        tb = "".join(format_exception(*exit_watcher.error))
        msg_err = f"{msg_err}\n\n{tb}"
        send_mail(subject=subject + " - Failure", msg=msg_err, **mail_kwargs)


@parse_config
class measure_time:
    """Context for timing a code block.

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
        name: Optional[str] = None,
        cpu: bool = False,
        logger: logging.Logger = logger,
    ):
        self.name = name or ""
        self.cpu = cpu
        self.logger = logger

    def __enter__(self):  # noqa: D105
        self.start = time.perf_counter()
        self.start_cpu = time.process_time()
        self.logger.info(f"Started process {self.name}.")
        return

    def __exit__(self, *args, **kwargs):  # noqa: D105
        elapsed = time.perf_counter() - self.start
        elapsed_cpu = time.process_time() - self.start_cpu
        occ = elapsed_cpu / elapsed
        s = f"Process {self.name} done in {elapsed:.02f} s"
        if self.cpu:
            s += f" and used {elapsed_cpu:.02f} of cpu time ({occ:.1%} % occupancy)."

        self.logger.info(s)


class TimeoutException(Exception):
    """An exception raised with a timeout occurs."""

    def __init__(self, seconds: int, task: str = "", **kwargs):
        self.msg = f"Task {task} timed out after {seconds} seconds"
        super().__init__(self.msg, **kwargs)


@contextmanager
def timeout(seconds: int, task: str = ""):
    """Timeout context manager.

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

        def _timeout_handler(signum, frame):
            raise TimeoutException(seconds, task)

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def skippable(seconds: int = 2, task: str = "", logger: logging.Logger = None):
    """Skippable context manager.

    When CTRL-C (SIGINT, KeyboardInterrupt) is sent within the context,
    this catches it, prints to the log and gives a timeout during which a subsequent
    interruption will stop the script. Otherwise, the context exits normally.

    This is meant to be used within a loop so we can skip some iterations:

    >>> for i in iterable:
    >>>    with skippable(2, i):
    >>>         ... skippable code ...

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
