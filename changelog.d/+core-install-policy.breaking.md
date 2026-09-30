A missing optional dependency now raises by default. `kornia_config.lazyloader.installation_mode` used to default to
`"ask"`, which asked on stdin whether to install the package: without an interactive stdin (a CI job, a DataLoader
worker) that raised `EOFError` instead of `ImportError`, and on an open stdin that nobody writes to it blocked. The
default is now `"raise"`: an `ImportError` naming `pip install "kornia[<extra>]"`. `"ask"` is opt-in and raises the same
`ImportError` when stdin is not a terminal; its "[A]ll" answer, which used to apply to one loader, now switches the
process to `"auto"`. `"auto"` used to run `pip install -U` with the module's import name and did not check pip's exit
status; it now installs the declared kornia extra (`pip install "kornia[<extra>]"`, without `-U`) instead of the import
name, raises `ImportError` if pip fails (without running pip again for that loader), and installs nothing for a loader
that declares no extra. The new `KORNIA_INSTALLATION_MODE` environment variable sets the mode when kornia is imported
(`raise`, `ask` or `auto`, any case; another value raises `ValueError`). The undocumented `LazyLoader.auto_install`
class switch is removed; set the mode to `"auto"` instead. `InstallationMode` members are now hashable and compare
equal only to their upper-case values (`InstallationMode.ASK == "ASK"`): `==` used to ignore case while `!=` did not,
so `InstallationMode.ASK == "ask"` and `InstallationMode.ASK != "ask"` were both true. The `installation_mode` setter
still accepts any case.
