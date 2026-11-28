# Making a release

A release has three files: `tpu_app` for Windows and for Linux, built by CI,
and the bitstreams, built with Vivado by hand (CI has no Vivado).

| File | Made by | Content |
|---|---|---|
| `tpu_app-windows-x64.zip` | CI (`.github/workflows/release.yml`) | `tpu_app.exe` with Qt and the MinGW runtime, `demo/` (models), `fpga/program.tcl` |
| `tpu_app-linux-x86_64.AppImage` | CI | the same, as one executable file (glibc 2.39 or newer: Ubuntu 24.04, Debian 13, Fedora 40...) |
| `bitstreams.zip` | you | `bitstreams/<size>/BoardTop.bit` and `config.json` for 2x2, 4x4 and 8x8 |

CI also builds both packages, and runs the host tests, on every pull request
that touches `host/`; they are under the run's artifacts.

## Steps

1. **Build the bitstreams** of the commit you release, from the repository root:

   ```
   sbt "runMain tpu.board.EmitBoard 2x2"
   vivado -mode batch -source fpga/build.tcl -tclargs 2x2
   ```

   and the same for `4x4` and `8x8`.

2. **Pack them** in the layout the app looks for. PowerShell:

   ```powershell
   foreach ($c in "2x2", "4x4", "8x8") {
       New-Item -ItemType Directory -Force "bitstreams/$c" | Out-Null
       Copy-Item "build/$c/BoardTop.bit", "build/$c/config.json" "bitstreams/$c/"
   }
   Compress-Archive -Path bitstreams -DestinationPath bitstreams.zip -Force
   ```

   or bash:

   ```
   for c in 2x2 4x4 8x8; do mkdir -p bitstreams/$c; cp build/$c/BoardTop.bit build/$c/config.json bitstreams/$c/; done
   zip -r bitstreams.zip bitstreams
   ```

3. **Tag and push**: `git tag v1.0.0`, then `git push origin v1.0.0`. CI builds both
   packages and creates a **draft** release with them.

4. **Attach the bitstreams** to the draft (`gh release upload v1.0.0 bitstreams.zip`
   or the release page), check the notes, and publish.

## Where the app looks for its files

The app finds its models and `program.tcl` next to its executable, and opens
**Program FPGA...** on a `bitstreams/` folder beside the app (beside the
`.AppImage` file on Linux). Built from the source tree, it falls back to
`host/cpp/demo/`, `fpga/program.tcl` and `build/`. The lookup is in
`host/cpp/gui/paths.cpp`.
