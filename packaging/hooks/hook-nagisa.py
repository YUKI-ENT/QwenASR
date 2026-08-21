"""PyInstaller hook for nagisa's package-relative legacy imports."""

from PyInstaller.utils.hooks import collect_all


datas, binaries, hiddenimports = collect_all("nagisa")

# nagisa adds its package directory to sys.path and imports modules such as
# prepro by their top-level names. Keep its Python sources on disk so those
# imports work in the frozen application as they do in a normal installation.
module_collection_mode = {"nagisa": "py"}
