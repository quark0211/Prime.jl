# Prime.jl

Prime.jl is a Julia package (under active development) hosted on GitHub.  
It is installable directly from this repository (no General registry registration required).

---

## 1) Prerequisites: Install Julia

### Recommended Julia version
Prime.jl currently targets **Julia 1.11.x** (your `Project.toml` compat is pinned around Julia 1.11 stdlibs and package versions).

### Install options
**Option A — Official download (GUI installers / binaries)**  
Download Julia from the official downloads page:  :contentReference[oaicite:0]{index=0}

**Option B — Platform-specific installation guide**  
If you want step-by-step platform instructions (Windows/macOS/Linux): :contentReference[oaicite:1]{index=1}

> Quick check after installation:
> ```bash
> julia --version
> ```

---

## 2) Install Prime.jl (direct from GitHub)

Open Julia and run:

```julia
using Pkg
Pkg.develop(PackageSpec(url="https://github.com/Li-shiyue/Prime.jl.git"))
using Prime
