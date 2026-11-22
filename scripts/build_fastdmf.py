"""Build the native C++ pybind11 extension for canonical FastDMF (concog/fastdmf).

Usage:
    .venv/bin/python scripts/build_fastdmf.py

Requires g++ (C++17) and pybind11. Automatically clones https://gitlab.com/concog/fastdmf.git
into third_party/fastdmf_gitlab if not already present, copies the public Schaefer-100 HCP
structural connectome into data/connectomes/, and builds:
    src/neurodyn_bench/models/_fastdmf_ext*.so
"""

from __future__ import annotations

import shutil
import subprocess
import sysconfig
from pathlib import Path

import pybind11


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    upstream = root / "third_party" / "fastdmf_gitlab"
    if not (upstream / "cpp" / "DMF.hpp").exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://gitlab.com/concog/fastdmf.git", str(upstream)],
            check=True,
        )

    # Copy bundled HCP Schaefer-100 structural connectome and upstream BSD-3 license
    conn_dir = root / "data" / "connectomes"
    conn_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        upstream / "python" / "fastdmf" / "DTI_fiber_consensus_HCP.csv",
        conn_dir / "DTI_fiber_consensus_HCP.csv",
    )
    shutil.copy2(upstream / "LICENSE", conn_dir / "LICENSE_FastDMF_BSD3.txt")

    src_cpp = root / "src" / "neurodyn_bench" / "models" / "_fastdmf_cpp.cpp"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_so = root / "src" / "neurodyn_bench" / "models" / f"_fastdmf_ext{ext_suffix}"

    py_inc = sysconfig.get_path("include")
    pyb_inc = pybind11.get_include()
    dmf_inc = str(upstream / "cpp")

    cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
        f"-I{py_inc}",
        f"-I{pyb_inc}",
        f"-I{dmf_inc}",
        str(src_cpp),
        "-o",
        str(out_so),
    ]
    print("Compiling FastDMF pybind11 extension:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built successfully: {out_so}")


if __name__ == "__main__":
    main()
