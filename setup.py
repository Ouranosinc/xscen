"""Custom installation process for xscen translations."""

from babel.messages.frontend import compile_catalog
from setuptools import setup
from setuptools.build_meta import *  # noqa: F403, F401
from setuptools.command.install import install


class InstallWithCompile(install):
    """Injection of the catalog compilation in the installation process."""

    def run(self):
        """Install the package, but compile the i18n catalogs first."""
        compiler = compile_catalog(self.distribution)
        option_dict = self.distribution.get_option_dict("compile_catalog")
        compiler.domain = [option_dict["domain"][1]]
        compiler.directory = option_dict["directory"][1]
        compiler.run()
        super().run()


setup(
    cmdclass={"install": InstallWithCompile},
    message_extractors={"xscen": [("**.py", "python", None)]},
)
