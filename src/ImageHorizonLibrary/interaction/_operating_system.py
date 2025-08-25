# -*- coding: utf-8 -*-
"""Operating system interaction keywords."""

import shlex
import subprocess

from ..errors import OSException


class _OperatingSystem(object):
    """Mixin providing application launch and termination keywords."""

    def launch_application(self, app, alias=None):
        """Launch an external application as a new process.

        Parameters
        ----------
        app : str
            Command used to start the application. The string is split using
            :func:`shlex.split` and passed to :class:`subprocess.Popen`.
            It should therefore match the command you would execute on the
            command line. On Windows, enclose paths containing spaces in
            double quotes.
        alias : str, optional
            Name used to reference the started process later. If omitted, an
            incremental alias is generated automatically.

        Returns
        -------
        str
            The alias associated with the launched application. This alias can
            be used with :py:meth:`terminate_application`.

        Examples
        --------
        | Launch Application | "C:\\my folder\\myprogram.exe" |
        | Launch Application | myprogram.exe | arg1 | arg2 |
        | Launch Application | myprogram.exe | alias=myprog |
        | Launch Application | myprogram.exe | arg1 | arg2 | alias=myprog |
        """
        if not alias:
            alias = str(len(self.open_applications))
        process = subprocess.Popen(shlex.split(app))
        self.open_applications[alias] = process
        return alias

    def terminate_application(self, alias=None):
        """Terminate a process started with :py:meth:`launch_application`.

        Parameters
        ----------
        alias : str, optional
            Alias of the process to terminate. If omitted, the most recently
            launched application is terminated.

        Returns
        -------
        None

        Raises
        ------
        OSException
            If the provided alias is invalid or no applications have been
            launched.
        """
        if alias and alias not in self.open_applications:
            raise OSException('Invalid alias "%s".' % alias)
        process = self.open_applications.pop(alias, None)
        if not process:
            try:
                _, process = self.open_applications.popitem()
            except KeyError:
                raise OSException(
                    "`Terminate Application` called without "
                    "`Launch Application` called first."
                )
        process.terminate()
