# vk-zig-matmul

Standalone Bazel/Zig Vulkan compute matmul prototype.

The runnable shader path is `//shaders:matmul_zig_spv`. Bazel invokes the
pinned Zig compiler on `shaders/matmul_kernel.zig`, then runs a tiny Zig
post-processor to add Vulkan-required SPIR-V decorations that Zig 0.16.0 does
not emit yet. There is no checked-in SPIR-V blob and no `fallback` CLI mode.
The build pins Vulkan-Headers for reproducible API inputs, but the host runner
currently uses a small hand-written Vulkan ABI shim because
`@cImport`/translate-c under `rules_zig` 0.12.2 + Zig 0.16.0 fails in Bazel's
sanitized action environment with `AppDataDirUnavailable`. Runtime Vulkan is
loaded dynamically from the host `libvulkan.so.1`.

Build:

```bash
bazel build //...
```

List Vulkan devices:

```bash
bazel run //:vk_matmul -- --list-devices
```

Build the Zig SPIR-V shader target:

```bash
bazel build //shaders:matmul_zig_spv
```

Run validation:

```bash
bazel run //:vk_matmul -- --shader=zig --m=16 --n=17 --k=19
bazel run //:vk_matmul -- --shader=zig --m=64 --n=64 --k=64
```

The current shader source avoids three Zig SPIR-V backend limitations:

- `zig build-obj` is used instead of `zig build-exe` to avoid `std/start.zig`.
- Fixed-size storage-buffer arrays are used instead of many-item logical
  pointers.
- `tools/patch_spv.zig` adds `LocalSize`, descriptor set/binding, and block
  decorations after Zig emits the module.
