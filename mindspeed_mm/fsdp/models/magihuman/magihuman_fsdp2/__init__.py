"""MagiHuman FSDP2 model plugin.

Importing this package (which `import_plugin` does via `pkgutil.walk_packages`)
triggers `@model_register.register("magihuman")` in `modeling_magihuman`.

We import the modeling module eagerly here so registration fires regardless of
walk order or whether the directory is treated as a namespace vs. regular
package. This module is intentionally side-effect-light: the heavy upstream DiT
import is deferred to build time inside `modeling_magihuman` (see the lazy
`_import_dit()` helper there), so plugin import succeeds even before the
upstream `inference/` package has been vendored in.
"""

from mindspeed_mm.fsdp.models.magihuman.magihuman_fsdp2 import modeling_magihuman
