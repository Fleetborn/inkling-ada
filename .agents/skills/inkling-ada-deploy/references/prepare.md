# Prepare the pinned build and model

## Choose verified inputs

- Project checkout at [Fleetborn/inkling-ada](https://github.com/Fleetborn/inkling-ada). Reuse a checkout with `serve.sh` and `patches/ninfer-sm89-tensor-qk.patch`, or clone it into an unused working directory in the user's cache. Record the project revision.
- Patched engine. Reuse a known build with a receipt identifying the pinned upstream revision, the project patch, the SM89 target, and the binary SHA-256. Otherwise build from source below.
- Model. The measured UD-IQ4_XS NInfer file is 16,494,910,976 bytes. Its SHA-256 is `a4a1bbfbdc1150cecc9df2bc12ae87c076330bff81579e243e7875115329bb54`. Verify a supplied copy or use the pinned model download. [Model repository](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3)

## Fetch missing model weights

Use a user-owned cache with space for the file and build. Resume an interrupted download using the partial file. Check the SHA-256 before renaming it to the final model filename.

```bash
set -e
filename=Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
sha=a4a1bbfbdc1150cecc9df2bc12ae87c076330bff81579e243e7875115329bb54
revision=50d6ad60cfcd170633ead778e3a09958fc1e7887
size=16494910976
mkdir -p "$cache/models"
cd "$cache/models"
candidate="$filename"
if [[ ! -f "$filename" ]]; then
  candidate="$filename.part"
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
if [[ "$candidate" != "$filename" ]]; then mv -- "$candidate" "$filename"; fi
model="$cache/models/$filename"
```

A full-sized partial is verified directly. Handle any digest mismatch in the skill's own download cache and preserve user-owned model files.

## Build the SM89 engine

Use a compatible Linux CUDA toolchain. CUDA 12.9 was used for the published benchmark. The upstream native build requires CMake 3.28 or newer, Ninja, a CUDA-supported C++20 compiler, libcurl and zlib development files, and its media build tools. Use the pinned upstream [native build script](https://github.com/iamwavecut/ninfer-all/blob/18029c8e0ce2555bb499e04be24d7a4e9c12a5bc/scripts/build-native.sh) to fetch and verify the media archives and configure the runtime. Review the exact [upstream build dependencies](https://github.com/iamwavecut/ninfer-all/blob/18029c8e0ce2555bb499e04be24d7a4e9c12a5bc/Dockerfile) if a prerequisite is missing.

Set `repo` and `cache` to absolute paths on the deployment host. Set `cuda_root` to the directory containing `bin/nvcc`. Use a fresh source directory so the scalar experiment and tensor-core patch never stack.

```bash
set -e
revision=18029c8e0ce2555bb499e04be24d7a4e9c12a5bc
patch="$repo/patches/ninfer-sm89-tensor-qk.patch"
test -x "$cuda_root/bin/nvcc"
mkdir -p "$cache"
src="$(mktemp -d "$cache/ninfer-sm89.XXXXXX")"
git -C "$src" init --quiet
git -C "$src" fetch --depth=1 https://github.com/iamwavecut/ninfer-all.git "$revision"
git -C "$src" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$src" rev-parse HEAD)" = "$revision"
git -C "$src" apply --check --whitespace=error "$patch"
git -C "$src" apply --whitespace=error "$patch"
git -C "$src" diff --check

export NINFER_CUDA_PATH="$cuda_root"
export NINFER_CUDA_ARCH=89 NINFER_TESTS=OFF NINFER_MULTICALL=OFF
export NINFER_BUILD_DIR="$src/build" NINFER_JOBS="${NINFER_JOBS:-4}"
bash "$src/scripts/build-native.sh" configure
bash "$src/scripts/build-native.sh" target ninfer-serve
engine="$NINFER_BUILD_DIR/apps/ninfer-serve"
"$engine" --version
printf 'upstream %s\narchitecture SM89\n' "$revision" > "$src/build-receipt.txt"
sha256sum "$patch" "$engine" >> "$src/build-receipt.txt"
```

If a shared library is missing, inspect `ldd "$engine"` and use the matching CUDA runtime library path when invoking the launcher. On a host without a suitable native toolchain, use an authorized compatible build environment or report the specific missing prerequisite. Keep the source build pinned even when upstream release downloads have rotated.

For a user-selected stock deployment, use an Inkling-capable upstream Linux release or an existing compatible binary. Verify the release checksums and ensure the archive matches the target distro and CUDA driver. Keep the runtime directories beside the executable. [Upstream Linux archive guide](https://github.com/iamwavecut/ninfer-all/blob/18029c8e0ce2555bb499e04be24d7a4e9c12a5bc/docs/release-archive-linux.md)
