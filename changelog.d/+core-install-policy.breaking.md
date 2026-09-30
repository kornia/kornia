A missing optional dependency now raises by default. `kornia_config.lazyloader.installation_mode` used to default to
`"ask"`, which asked on stdin whether to install the package: without an interactive stdin (a CI job, a DataLoader
worker) that raised `EOFError` instead of `ImportError`, and on an open stdin that nobody writes to it blocked. The
default is now `"raise"`: an `ImportError` naming `pip install "kornia[<extra>]"`. `"ask"` is opt-in and raises the same
`ImportError` when stdin is not a terminal; its "[A]ll" answer, which used to apply to one loader, now switches the
process to `"auto"`. `"auto"` used to install the loader's import name with `-U`, even where it is not a package name,
and did not check pip's exit status; it now installs the declared kornia extra instead of the import name
(`pip install "kornia[<extra>]"`, for example `kornia[image]`, without `-U`), raises `ImportError` if pip fails (without
running pip again for that loader), and installs nothing for a loader that declares no extra. The new
`KORNIA_INSTALLATION_MODE` environment variable sets the mode when kornia is imported (`raise`, `ask` or `auto`, any
case); another value does not stop `import kornia`, but raises a `ValueError` naming the variable and the valid modes
the first time a missing optional package is handled. The undocumented `LazyLoader.auto_install` class switch is
removed; set the mode to `"auto"` instead. `InstallationMode` members are now hashable and compare equal only to their
upper-case values (`InstallationMode.ASK == "ASK"`): `==` used to ignore case while `!=` did not, so
`InstallationMode.ASK == "ask"` and `InstallationMode.ASK != "ask"` were both true. The `installation_mode` setter
still accepts any case.
