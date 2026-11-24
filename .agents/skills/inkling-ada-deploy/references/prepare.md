# Prepare the pinned build and model

## Choose verified inputs

- Project checkout at [Fleetborn/inkling-ada](https://github.com/Fleetborn/inkling-ada). Set `repo` to its absolute path on the target host. Record the project revision.
- Engine. Use the clean upstream revision below, built for SM89. Verify a prepared build against its receipt or build it using the native toolchain.
- Model. Use the 16,595,277,824-byte UD-IQ4_XS artifact with the mainline DSpark drafter. Verify the pinned digest before use. [Model revision](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3/tree/0baae7fdb07506ec285bd334f2671a889aa890c0)

## Fetch missing model weights

Use a user-owned cache with space for the file and build. Resume an interrupted download using the partial file. Check the SHA-256 before renaming it to the final model filename.

```bash
set -eu
cache="${XDG_CACHE_HOME:-$HOME/.cache}/inkling-ada"
filename=Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
sha=d96986947048ed3ac643da0ed6c389e8ab821f66d10e2abf1f4347224b4180db
revision=0baae7fdb07506ec285bd334f2671a889aa890c0
size=16595277824
mkdir -p "$cache/models/$revision"
model="$cache/models/$revision/$filename"
candidate="$model"
if [[ ! -f "$model" ]]; then
  candidate="$model.part"
  bytes=0
  if [[ -f "$candidate" ]]; then bytes="$(stat -c %s -- "$candidate")"; fi
  if (( bytes > size )); then
    printf 'Partial download exceeds the pinned model size: %s\n' "$candidate" >&2
    exit 1
  fi
  if (( bytes < size )); then
    curl --fail --location --retry 2 --continue-at - --output "$candidate" \
      "https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3/resolve/$revision/$filename"
  fi
fi
printf '%s  %s\n' "$sha" "$candidate" | sha256sum --check -
if [[ "$candidate" != "$model" ]]; then mv -- "$candidate" "$model"; fi
```

A full-sized partial is verified directly. Handle any digest mismatch in the skill's own download cache and preserve user-owned model files.

## Build the SM89 engine

Use a compatible Linux CUDA toolchain. CUDA 12.9.86 was used for the published SM89 benchmark. The upstream native build requires CMake 3.28 or newer, Ninja, a CUDA-supported C++20 compiler, libcurl and zlib development files, and its media build tools. The pinned upstream [native build script](https://github.com/iamwavecut/ninfer-all/blob/7894c119f638eb6cae52e4b3e8efc53ebaad216b/scripts/build-native.sh) fetches and verifies the media archives. Review the exact [upstream build dependencies](https://github.com/iamwavecut/ninfer-all/blob/7894c119f638eb6cae52e4b3e8efc53ebaad216b/Dockerfile) if a prerequisite is missing.

Set `NINFER_CUDA_PATH` if the toolkit is outside `/usr/local/cuda`. The recipe reuses the clean pinned checkout and its incremental build on retries. If an interrupted checkout is incomplete or has local changes, move its revision-specific engine directory aside to preserve it before retrying.

```bash
set -eu
cache="${XDG_CACHE_HOME:-$HOME/.cache}/inkling-ada"
revision=7894c119f638eb6cae52e4b3e8efc53ebaad216b
root="$cache/engines/$revision"
src="$root/src"
mkdir -p "$root"
if [[ ! -e "$src" ]]; then git init --quiet "$src"; fi
test -d "$src/.git"
test -z "$(git -C "$src" status --porcelain)"
if ! git -C "$src" rev-parse --verify HEAD >/dev/null 2>&1; then
  git -C "$src" fetch --depth=1 https://github.com/iamwavecut/ninfer-all.git "$revision"
  git -C "$src" checkout --quiet --detach FETCH_HEAD
fi
test "$(git -C "$src" rev-parse HEAD)" = "$revision"

export NINFER_CUDA_PATH="${NINFER_CUDA_PATH:-/usr/local/cuda}"
test -x "$NINFER_CUDA_PATH/bin/nvcc"
export NINFER_CUDA_ARCH=89 NINFER_TESTS=OFF NINFER_MULTICALL=OFF
export NINFER_BUILD_DIR="$root/build" NINFER_MEDIA_SOURCES="$root/media" NINFER_JOBS="${NINFER_JOBS:-4}"
bash "$src/scripts/build-native.sh" configure
bash "$src/scripts/build-native.sh" target ninfer-serve
engine="$NINFER_BUILD_DIR/apps/ninfer-serve"
"$engine" --version
printf 'upstream %s\narchitecture SM89\n' "$revision" > "$root/build-receipt.txt"
sha256sum "$engine" >> "$root/build-receipt.txt"
```

If a shared library is missing, inspect `ldd "$engine"` and use the matching CUDA runtime library path when invoking the launcher. On a host without a suitable native toolchain, use an authorized compatible build environment or report the specific missing prerequisite. Keep the source build pinned even when upstream release downloads have rotated.

From the Inkling Ada checkout, run `python3 .agents/skills/inkling-ada-deploy/scripts/start_server.py --repo "$PWD"`. The launcher resolves the revision-specific build and model prepared above. For verified files elsewhere, supply their absolute paths with `--engine` and `--model`.
