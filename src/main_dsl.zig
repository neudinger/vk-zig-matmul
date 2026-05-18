const std = @import("std");
const matmul_coop_bf16_opt_spv = @import("matmul_coop_bf16_opt_spv");
const matmul_coop_bf16_spv = @import("matmul_coop_bf16_spv");
const matmul_coop_f16_opt_spv = @import("matmul_coop_f16_opt_dsl_spv");
const matmul_coop_f16_spv = @import("matmul_coop_f16_spv");
const matmul_coop_shared_f16_spv = @import("matmul_coop_shared_f16_spv");
const matmul_nvcoop2_bf16_spv = @import("matmul_nvcoop2_bf16_spv");
const matmul_nvcoop2_f16_spv = @import("matmul_nvcoop2_f16_spv");
const matmul_nvcoop2_square_f16_frontend_spv = @import("matmul_nvcoop2_square_f16_frontend_spv");
const matmul_nvcoop2_square_f16_spv = @import("matmul_nvcoop2_square_f16_spv");
const matmul_nvcoop2_wide_f16_spv = @import("matmul_nvcoop2_wide_f16_spv");
const matmul_zig_spv = @import("matmul_zig_dsl_spv");
const zvk = @import("vulkan");

const log = std.log.scoped(.vk_matmul);

const CLOCK_MONOTONIC: c_int = 1;
const Timespec = extern struct {
    tv_sec: isize,
    tv_nsec: isize,
};
extern fn clock_gettime(clk_id: c_int, tp: *Timespec) callconv(.c) c_int;

const ShaderMode = enum {
    zig,
    coop_bf16,
    coop_f16,
    coop_bf16_opt,
    coop_f16_opt,
    coop_shared_f16,
    nvcoop2_bf16,
    nvcoop2_f16,
    nvcoop2_wide_f16,
    nvcoop2_square_f16,
    nvcoop2_square_f16_frontend,

    fn isCoop(self: ShaderMode) bool {
        return self != .zig;
    }

    fn isNvCoop2(self: ShaderMode) bool {
        return switch (self) {
            .nvcoop2_bf16, .nvcoop2_f16, .nvcoop2_wide_f16, .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => true,
            .zig, .coop_bf16, .coop_f16, .coop_bf16_opt, .coop_f16_opt, .coop_shared_f16 => false,
        };
    }

    fn name(self: ShaderMode) []const u8 {
        return switch (self) {
            .zig => "zig",
            .coop_bf16 => "coop",
            .coop_f16 => "coop-f16",
            .coop_bf16_opt => "coop-opt",
            .coop_f16_opt => "coop-f16-opt",
            .coop_shared_f16 => "coop-shared-f16",
            .nvcoop2_bf16 => "nvcoop2",
            .nvcoop2_f16 => "nvcoop2-f16",
            .nvcoop2_wide_f16 => "nvcoop2-wide-f16",
            .nvcoop2_square_f16 => "nvcoop2-square-f16",
            .nvcoop2_square_f16_frontend => "nvcoop2-square-f16-frontend",
        };
    }

    fn isBf16(self: ShaderMode) bool {
        return switch (self) {
            .coop_bf16, .coop_bf16_opt, .nvcoop2_bf16 => true,
            .zig, .coop_f16, .coop_f16_opt, .coop_shared_f16, .nvcoop2_f16, .nvcoop2_wide_f16, .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => false,
        };
    }

    fn isF16(self: ShaderMode) bool {
        return switch (self) {
            .coop_f16, .coop_f16_opt, .coop_shared_f16, .nvcoop2_f16, .nvcoop2_wide_f16, .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => true,
            .zig, .coop_bf16, .coop_bf16_opt, .nvcoop2_bf16 => false,
        };
    }

    fn inputComponentType(self: ShaderMode) zvk.ComponentTypeKHR {
        return switch (self) {
            .zig => zvk.ComponentTypeKHR.float32_khr,
            .coop_bf16, .coop_bf16_opt, .nvcoop2_bf16 => zvk.ComponentTypeKHR.bfloat16_khr,
            .coop_f16, .coop_f16_opt, .coop_shared_f16, .nvcoop2_f16, .nvcoop2_wide_f16, .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => zvk.ComponentTypeKHR.float16_khr,
        };
    }

    fn matrixTileN(self: ShaderMode) usize {
        return switch (self) {
            .zig => 16,
            .coop_bf16, .coop_bf16_opt => 16,
            .coop_f16, .coop_f16_opt, .coop_shared_f16 => 8,
            .nvcoop2_bf16, .nvcoop2_f16, .nvcoop2_wide_f16, .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => 64,
        };
    }

    fn outputTileM(self: ShaderMode) usize {
        return switch (self) {
            .zig, .coop_bf16, .coop_f16, .coop_bf16_opt, .coop_f16_opt => 16,
            .coop_shared_f16 => 64,
            .nvcoop2_bf16, .nvcoop2_f16, .nvcoop2_wide_f16 => 64,
            .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => 128,
        };
    }

    fn outputTileN(self: ShaderMode) usize {
        return switch (self) {
            .zig => 16,
            .coop_bf16 => 16,
            .coop_f16 => 8,
            .coop_bf16_opt, .coop_f16_opt => 32,
            .coop_shared_f16 => 64,
            .nvcoop2_bf16, .nvcoop2_f16 => 64,
            .nvcoop2_wide_f16, .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => 128,
        };
    }
};

const TimingMode = enum {
    gpu_timestamp,
    submit_cpu,
};

const Options = struct {
    shader: ShaderMode = .zig,
    m: usize = 64,
    n: usize = 64,
    k: usize = 64,
    iters: usize = 50,
    warmup: usize = 5,
    timing: TimingMode = .gpu_timestamp,
    skip_validation: bool = false,
    list_devices: bool = false,
    device_index: ?usize = null,
    device_substr: ?[]const u8 = null,
};

const TimingResult = struct {
    avg_ns: f64,
    avg_gpu_ns: ?f64 = null,
    avg_batch_cpu_ns: ?f64 = null,
};

const PushConstants = extern struct {
    m: u32,
    n: u32,
    k: u32,
    a_stride: u32,
    b_stride: u32,
    c_stride: u32,
};

const Buffer = struct {
    handle: zvk.Buffer,
    memory: zvk.DeviceMemory,
    mapped: ?[*]u8,
    byte_len: usize,

    fn bytes(self: Buffer) usize {
        return self.byte_len;
    }

    fn slice(self: Buffer, comptime T: type) []T {
        const ptr: [*]T = @ptrCast(@alignCast(self.mapped.?));
        return ptr[0 .. self.byte_len / @sizeOf(T)];
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);

    const opts = try parseArgs(args);
    if (opts.device_index != null and opts.device_substr != null) {
        log.err("use either --device=<index> or --device-substr=<text>, not both", .{});
        return error.InvalidArgument;
    }
    if (opts.m == 0 or opts.n == 0 or opts.k == 0) return error.InvalidDimensions;
    if (opts.iters == 0) return error.InvalidIterations;
    if (opts.m > std.math.maxInt(u32) or opts.n > std.math.maxInt(u32) or opts.k > std.math.maxInt(u32)) {
        return error.InvalidDimensions;
    }
    if (opts.shader.isCoop()) validateCoopDimensions(opts);

    var vk = try Vulkan.open();
    defer vk.close();

    const instance = try vk.createInstance();
    defer vk.instance.dispatch.vkDestroyInstance.?(instance, null);

    const selected = try selectPhysicalDevice(&vk, allocator, instance, opts.device_index, opts.device_substr, opts.list_devices);
    if (opts.list_devices) return;

    if (opts.shader.isCoop()) {
        requireCoopMatrixProperty(&vk, allocator, selected.physical_device, opts.shader) catch |err| switch (err) {
            error.RequiredDeviceExtensionMissing,
            error.RequiredDeviceFeatureMissing,
            error.RequiredCoopMatrixPropertyMissing,
            => std.process.exit(2),
            else => return err,
        };
    }

    var device = createDevice(&vk, allocator, selected.physical_device, selected.queue_family, opts.shader) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing,
        error.RequiredDeviceFeatureMissing,
        => std.process.exit(2),
        else => return err,
    };
    defer device.deinit();

    try runMatmul(&vk, &device, opts);
}

fn parseArgs(args: []const []const u8) !Options {
    var opts: Options = .{};
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--list-devices")) {
            opts.list_devices = true;
        } else if (std.mem.eql(u8, arg, "--skip-validation")) {
            opts.skip_validation = true;
        } else if (std.mem.startsWith(u8, arg, "--shader=")) {
            const value = arg["--shader=".len..];
            if (std.mem.eql(u8, value, "zig")) {
                opts.shader = .zig;
            } else if (std.mem.eql(u8, value, "coop") or std.mem.eql(u8, value, "coop-bf16")) {
                opts.shader = .coop_bf16;
            } else if (std.mem.eql(u8, value, "coop-f16")) {
                opts.shader = .coop_f16;
            } else if (std.mem.eql(u8, value, "coop-opt") or std.mem.eql(u8, value, "coop-bf16-opt")) {
                opts.shader = .coop_bf16_opt;
            } else if (std.mem.eql(u8, value, "coop-f16-opt")) {
                opts.shader = .coop_f16_opt;
            } else if (std.mem.eql(u8, value, "coop-shared-f16")) {
                opts.shader = .coop_shared_f16;
            } else if (std.mem.eql(u8, value, "nvcoop2") or std.mem.eql(u8, value, "nvcoop2-bf16")) {
                opts.shader = .nvcoop2_bf16;
            } else if (std.mem.eql(u8, value, "nvcoop2-f16")) {
                opts.shader = .nvcoop2_f16;
            } else if (std.mem.eql(u8, value, "nvcoop2-wide-f16")) {
                opts.shader = .nvcoop2_wide_f16;
            } else if (std.mem.eql(u8, value, "nvcoop2-square-f16")) {
                opts.shader = .nvcoop2_square_f16;
            } else if (std.mem.eql(u8, value, "nvcoop2-square-f16-frontend")) {
                opts.shader = .nvcoop2_square_f16_frontend;
            } else {
                log.err("unknown shader mode: {s}", .{value});
                return error.InvalidShader;
            }
        } else if (std.mem.startsWith(u8, arg, "--m=")) {
            opts.m = try std.fmt.parseInt(usize, arg["--m=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--n=")) {
            opts.n = try std.fmt.parseInt(usize, arg["--n=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--k=")) {
            opts.k = try std.fmt.parseInt(usize, arg["--k=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--iters=")) {
            opts.iters = try std.fmt.parseInt(usize, arg["--iters=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            opts.warmup = try std.fmt.parseInt(usize, arg["--warmup=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--timing=")) {
            const value = arg["--timing=".len..];
            if (std.mem.eql(u8, value, "gpu") or std.mem.eql(u8, value, "gpu-timestamp")) {
                opts.timing = .gpu_timestamp;
            } else if (std.mem.eql(u8, value, "submit-cpu")) {
                opts.timing = .submit_cpu;
            } else {
                log.err("unknown timing mode: {s}", .{value});
                return error.InvalidArgument;
            }
        } else if (std.mem.startsWith(u8, arg, "--device=")) {
            opts.device_index = try std.fmt.parseInt(usize, arg["--device=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--device-substr=")) {
            opts.device_substr = arg["--device-substr=".len..];
        } else {
            log.err("unknown argument: {s}", .{arg});
            return error.InvalidArgument;
        }
    }
    return opts;
}

const Vulkan = struct {
    lib: std.DynLib,
    getInstanceProcAddr: zvk.PfnGetInstanceProcAddr,
    base: zvk.BaseWrapper,
    instance: zvk.InstanceWrapper,

    fn open() !Vulkan {
        var lib = try std.DynLib.open("libvulkan.so.1");
        errdefer lib.close();

        const gip = lib.lookup(zvk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return error.SymbolNotFound;
        var base = zvk.BaseWrapper{ .dispatch = .{} };
        base.dispatch.vkGetInstanceProcAddr = gip;
        base.dispatch.vkCreateInstance = @ptrCast(gip(.null_handle, "vkCreateInstance") orelse return error.SymbolNotFound);

        return .{
            .lib = lib,
            .getInstanceProcAddr = gip,
            .base = base,
            .instance = undefined,
        };
    }

    fn close(self: *Vulkan) void {
        self.lib.close();
    }

    fn createInstance(self: *Vulkan) !zvk.Instance {
        var app: zvk.ApplicationInfo = .{
            .application_version = zvk.makeApiVersion(0, 0, 1, 0).toU32(),
            .engine_version = zvk.makeApiVersion(0, 0, 1, 0).toU32(),
            .api_version = zvk.API_VERSION_1_4.toU32(),
        };
        app.p_application_name = "vk-zig-matmul";
        app.p_engine_name = "none";

        var info: zvk.InstanceCreateInfo = .{};
        info.p_application_info = &app;

        const instance = try self.base.createInstance(&info, null);
        self.instance = zvk.InstanceWrapper.load(instance, self.getInstanceProcAddr);
        return instance;
    }
};

const SelectedDevice = struct {
    physical_device: zvk.PhysicalDevice,
    queue_family: u32,
};

fn selectPhysicalDevice(vk: *Vulkan, allocator: std.mem.Allocator, instance: zvk.Instance, device_index: ?usize, device_substr: ?[]const u8, list_only: bool) !SelectedDevice {
    var count: u32 = 0;
    try vkCheck(vk.instance.dispatch.vkEnumeratePhysicalDevices.?(instance, &count, null));
    if (count == 0) return error.NoVulkanDevice;
    if (device_index) |index| {
        if (index >= @as(usize, count)) {
            log.err("--device={d} is out of range; use --list-devices to see available Vulkan devices", .{index});
            std.process.exit(2);
        }
    }

    const devices = try allocator.alloc(zvk.PhysicalDevice, count);
    defer allocator.free(devices);
    try vkCheck(vk.instance.dispatch.vkEnumeratePhysicalDevices.?(instance, &count, devices.ptr));

    var selected: ?SelectedDevice = null;
    for (devices[0..count], 0..) |physical_device, i| {
        var props: zvk.PhysicalDeviceProperties = undefined;
        vk.instance.dispatch.vkGetPhysicalDeviceProperties.?(physical_device, &props);
        const name = deviceName(&props);
        const queue_family = findComputeQueueFamily(vk, allocator, physical_device) catch null;

        std.debug.print("device[{d}]: {s} type={d} api={d}.{d}.{d}{s}\n", .{
            i,
            name,
            @intFromEnum(deviceType(&props)),
            versionMajor(api_version(&props)),
            versionMinor(api_version(&props)),
            versionPatch(api_version(&props)),
            if (queue_family == null) " no-compute-queue" else "",
        });

        if (list_only or queue_family == null) continue;
        if (device_index) |index| {
            if (i != index) continue;
        }
        if (device_substr) |needle| {
            if (std.mem.indexOf(u8, name, needle) == null) continue;
        } else if (deviceType(&props) == zvk.PhysicalDeviceType.cpu) {
            continue;
        }
        if (selected == null) {
            selected = .{ .physical_device = physical_device, .queue_family = queue_family.? };
        }
    }

    if (list_only) return .{ .physical_device = devices[0], .queue_family = 0 };
    return selected orelse error.NoMatchingDevice;
}

fn findComputeQueueFamily(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: zvk.PhysicalDevice) !?u32 {
    var count: u32 = 0;
    vk.instance.dispatch.vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &count, null);
    if (count == 0) return null;

    const families = try allocator.alloc(zvk.QueueFamilyProperties, count);
    defer allocator.free(families);
    vk.instance.dispatch.vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &count, families.ptr);

    for (families[0..count], 0..) |family, i| {
        if (family.queue_flags.compute_bit) return @intCast(i);
    }
    return null;
}

fn validateCoopDimensions(opts: Options) void {
    if (opts.shader.isNvCoop2()) {
        if (opts.m % opts.shader.outputTileM() == 0 and opts.n % opts.shader.outputTileN() == 0 and opts.k % 16 == 0) return;
        log.err("--shader={s} requires m % {d} == 0, n % {d} == 0, and k % 16 == 0; use --shader=zig or KHR coop modes for other dimensions", .{ opts.shader.name(), opts.shader.outputTileM(), opts.shader.outputTileN() });
        std.process.exit(2);
    }

    const tile_n = opts.shader.outputTileN();
    if (opts.m % opts.shader.outputTileM() == 0 and opts.n % tile_n == 0 and opts.k % 16 == 0) return;
    log.err("--shader={s} requires m % 16 == 0, n % {d} == 0, and k % 16 == 0; use --shader=zig for arbitrary dimensions", .{ opts.shader.name(), tile_n });
    std.process.exit(2);
}

fn requireDeviceExtensions(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: zvk.PhysicalDevice, shader: ShaderMode) !void {
    if (!shader.isCoop()) return;

    var count: u32 = 0;
    try vkCheck(vk.instance.dispatch.vkEnumerateDeviceExtensionProperties.?(physical_device, null, &count, null));
    const extensions = try allocator.alloc(zvk.ExtensionProperties, count);
    defer allocator.free(extensions);
    try vkCheck(vk.instance.dispatch.vkEnumerateDeviceExtensionProperties.?(physical_device, null, &count, extensions.ptr));

    if (!hasDeviceExtension(extensions, zvk.extensions.khr_cooperative_matrix.name)) {
        log.err("{s} requires {s}, but the selected device does not advertise it", .{ shader.name(), zvk.extensions.khr_cooperative_matrix.name });
        return error.RequiredDeviceExtensionMissing;
    }
    if (shader.isNvCoop2() and !hasDeviceExtension(extensions, zvk.extensions.nv_cooperative_matrix_2.name)) {
        log.err("{s} requires {s}, but the selected device does not advertise it", .{ shader.name(), zvk.extensions.nv_cooperative_matrix_2.name });
        return error.RequiredDeviceExtensionMissing;
    }
    if (shader.isBf16() and !hasDeviceExtension(extensions, zvk.extensions.khr_shader_bfloat_16.name)) {
        log.err("{s} requires {s}, but the selected device does not advertise it", .{ shader.name(), zvk.extensions.khr_shader_bfloat_16.name });
        return error.RequiredDeviceExtensionMissing;
    }
}

fn hasDeviceExtension(extensions: []const zvk.ExtensionProperties, needle: []const u8) bool {
    for (extensions) |extension| {
        const name = std.mem.sliceTo(&extension.extension_name, 0);
        if (std.mem.eql(u8, name, needle)) return true;
    }
    return false;
}

fn requireCoopMatrixProperty(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: zvk.PhysicalDevice, shader: ShaderMode) !void {
    if (shader.isNvCoop2()) return requireNvCoop2Property(vk, allocator, physical_device, shader);

    const get_props = vk.instance.dispatch.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR orelse {
        log.err("Vulkan loader does not expose vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR", .{});
        return error.RequiredDeviceExtensionMissing;
    };

    var count: u32 = 0;
    try vkCheck(get_props(physical_device, &count, null));
    if (count == 0) {
        log.err("{s} requires a 16x8x16 subgroup cooperative-matrix property; selected device returned none", .{shader.name()});
        return error.RequiredCoopMatrixPropertyMissing;
    }

    const props = try allocator.alloc(zvk.CooperativeMatrixPropertiesKHR, count);
    defer allocator.free(props);
    for (props) |*prop| {
        prop.* = std.mem.zeroes(zvk.CooperativeMatrixPropertiesKHR);
        prop.s_type = .cooperative_matrix_properties_khr;
    }
    try vkCheck(get_props(physical_device, &count, props.ptr));

    const input_type = shader.inputComponentType();
    for (props[0..count]) |prop| {
        if (prop.m_size == 16 and prop.n_size == @as(u32, @intCast(shader.matrixTileN())) and prop.k_size == 16 and
            prop.a_type == input_type and prop.b_type == input_type and
            prop.c_type == zvk.ComponentTypeKHR.float32_khr and
            prop.result_type == zvk.ComponentTypeKHR.float32_khr and
            prop.saturating_accumulation == zvk.Bool32.false and
            prop.scope == zvk.ScopeKHR.subgroup_khr)
        {
            return;
        }
    }

    log.err("selected device lacks required {s} cooperative matrix property: M=16 N={d} K=16 A/B={d} C/Result=FLOAT32 scope=SUBGROUP", .{ shader.name(), shader.matrixTileN(), input_type });
    for (props[0..@min(count, 24)]) |prop| {
        log.err("  property: M={d} N={d} K={d} A={d} B={d} C={d} Result={d} sat={d} scope={d}", .{
            prop.m_size,
            prop.n_size,
            prop.k_size,
            prop.a_type,
            prop.b_type,
            prop.c_type,
            prop.result_type,
            prop.saturating_accumulation,
            prop.scope,
        });
    }
    return error.RequiredCoopMatrixPropertyMissing;
}

fn requireNvCoop2Property(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: zvk.PhysicalDevice, shader: ShaderMode) !void {
    var base_props: zvk.PhysicalDeviceProperties = undefined;
    vk.instance.dispatch.vkGetPhysicalDeviceProperties.?(physical_device, &base_props);
    if (vendorId(&base_props) != 0x10de) {
        log.err("{s} is NVIDIA-specific and requires vendorID 0x10de; selected vendorID is 0x{x}", .{ shader.name(), vendorId(&base_props) });
        return error.RequiredDeviceFeatureMissing;
    }

    var nv2_features: zvk.PhysicalDeviceCooperativeMatrix2FeaturesNV = std.mem.zeroes(zvk.PhysicalDeviceCooperativeMatrix2FeaturesNV);
    nv2_features.s_type = .physical_device_cooperative_matrix_2_features_nv;
    var khr_features: zvk.PhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(zvk.PhysicalDeviceCooperativeMatrixFeaturesKHR);
    khr_features.s_type = .physical_device_cooperative_matrix_features_khr;
    khr_features.p_next = &nv2_features;
    var features2: zvk.PhysicalDeviceFeatures2 = std.mem.zeroes(zvk.PhysicalDeviceFeatures2);
    features2.s_type = .physical_device_features_2;
    features2.p_next = &khr_features;
    vk.instance.dispatch.vkGetPhysicalDeviceFeatures2.?(physical_device, &features2);

    if (khr_features.cooperative_matrix != zvk.Bool32.true or
        nv2_features.cooperative_matrix_workgroup_scope != zvk.Bool32.true or
        nv2_features.cooperative_matrix_flexible_dimensions != zvk.Bool32.true or
        nv2_features.cooperative_matrix_tensor_addressing != zvk.Bool32.true or
        nv2_features.cooperative_matrix_block_loads != zvk.Bool32.true)
    {
        log.err("{s} requires KHR cooperative matrix plus NV coop2 workgroup scope, flexible dimensions, tensor addressing, and block loads", .{shader.name()});
        log.err("  features: khr={d} workgroup={d} flexible={d} tensor={d} block={d}", .{
            khr_features.cooperative_matrix,
            nv2_features.cooperative_matrix_workgroup_scope,
            nv2_features.cooperative_matrix_flexible_dimensions,
            nv2_features.cooperative_matrix_tensor_addressing,
            nv2_features.cooperative_matrix_block_loads,
        });
        return error.RequiredDeviceFeatureMissing;
    }

    var nv2_props: zvk.PhysicalDeviceCooperativeMatrix2PropertiesNV = std.mem.zeroes(zvk.PhysicalDeviceCooperativeMatrix2PropertiesNV);
    nv2_props.s_type = .physical_device_cooperative_matrix_2_properties_nv;
    var properties2: zvk.PhysicalDeviceProperties2 = std.mem.zeroes(zvk.PhysicalDeviceProperties2);
    properties2.s_type = .physical_device_properties_2;
    properties2.p_next = &nv2_props;
    vk.instance.dispatch.vkGetPhysicalDeviceProperties2.?(physical_device, &properties2);
    const required_max_dim = @max(shader.outputTileM(), shader.outputTileN());
    if (nv2_props.cooperative_matrix_workgroup_scope_max_workgroup_size < 256 or nv2_props.cooperative_matrix_flexible_dimensions_max_dimension < required_max_dim) {
        log.err("{s} requires NV coop2 maxWorkgroupSize >= 256 and maxDimension >= {d}; got maxWorkgroupSize={d} maxDimension={d}", .{
            shader.name(),
            required_max_dim,
            nv2_props.cooperative_matrix_workgroup_scope_max_workgroup_size,
            nv2_props.cooperative_matrix_flexible_dimensions_max_dimension,
        });
        return error.RequiredCoopMatrixPropertyMissing;
    }

    const get_props = vk.instance.dispatch.vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV orelse {
        log.err("Vulkan loader does not expose vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV", .{});
        return error.RequiredDeviceExtensionMissing;
    };

    var count: u32 = 0;
    try vkCheck(get_props(physical_device, &count, null));
    if (count == 0) {
        log.err("{s} requires NV cooperative matrix flexible-dimension properties; selected device returned none", .{shader.name()});
        return error.RequiredCoopMatrixPropertyMissing;
    }

    const props = try allocator.alloc(zvk.CooperativeMatrixFlexibleDimensionsPropertiesNV, count);
    defer allocator.free(props);
    for (props) |*prop| {
        prop.* = std.mem.zeroes(zvk.CooperativeMatrixFlexibleDimensionsPropertiesNV);
        prop.s_type = .cooperative_matrix_flexible_dimensions_properties_nv;
    }
    try vkCheck(get_props(physical_device, &count, props.ptr));

    const input_type = shader.inputComponentType();
    const tile_m: u32 = @intCast(shader.outputTileM());
    const tile_n: u32 = @intCast(shader.outputTileN());
    for (props[0..count]) |prop| {
        if (tile_m % prop.m_granularity == 0 and tile_n % prop.n_granularity == 0 and 16 % prop.k_granularity == 0 and
            prop.a_type == input_type and prop.b_type == input_type and
            prop.c_type == zvk.ComponentTypeKHR.float32_khr and
            prop.result_type == zvk.ComponentTypeKHR.float32_khr and
            prop.saturating_accumulation == zvk.Bool32.false and
            prop.scope == zvk.ScopeKHR.workgroup_khr and
            prop.workgroup_invocations == 256)
        {
            return;
        }
    }

    log.err("selected device lacks required {s} NV coop2 property: tile={d}x{d}x16 A/B={d} C/Result=FLOAT32 scope=WORKGROUP invocations=256", .{ shader.name(), tile_m, tile_n, input_type });
    for (props[0..@min(count, 32)]) |prop| {
        log.err("  property: Mgran={d} Ngran={d} Kgran={d} A={d} B={d} C={d} Result={d} sat={d} scope={d} invocations={d}", .{
            prop.m_granularity,
            prop.n_granularity,
            prop.k_granularity,
            prop.a_type,
            prop.b_type,
            prop.c_type,
            prop.result_type,
            prop.saturating_accumulation,
            prop.scope,
            prop.workgroup_invocations,
        });
    }
    return error.RequiredCoopMatrixPropertyMissing;
}

const Device = struct {
    handle: zvk.Device,
    fns: zvk.DeviceWrapper,
    queue: zvk.Queue,
    physical_device: zvk.PhysicalDevice,
    queue_family: u32,

    fn deinit(self: *Device) void {
        _ = self.fns.dispatch.vkDeviceWaitIdle.?(self.handle);
        self.fns.dispatch.vkDestroyDevice.?(self.handle, null);
    }
};

fn createDevice(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: zvk.PhysicalDevice, queue_family: u32, shader: ShaderMode) !Device {
    try requireDeviceExtensions(vk, allocator, physical_device, shader);

    var priorities = [_]f32{1.0};
    var queue_info: zvk.DeviceQueueCreateInfo = std.mem.zeroes(zvk.DeviceQueueCreateInfo);
    queue_info.s_type = .device_queue_create_info;
    queue_info.queue_family_index = queue_family;
    queue_info.queue_count = 1;
    queue_info.p_queue_priorities = priorities[0..].ptr;

    var device_info: zvk.DeviceCreateInfo = std.mem.zeroes(zvk.DeviceCreateInfo);
    device_info.s_type = .device_create_info;
    device_info.queue_create_info_count = 1;
    var queue_infos = [_]zvk.DeviceQueueCreateInfo{queue_info};
    device_info.p_queue_create_infos = queue_infos[0..].ptr;

    var extensions = [_][*:0]const u8{
        zvk.extensions.khr_cooperative_matrix.name,
        zvk.extensions.khr_shader_bfloat_16.name,
        zvk.extensions.nv_cooperative_matrix_2.name,
    };
    if (shader.isCoop()) {
        device_info.enabled_extension_count = if (shader.isNvCoop2()) (if (shader.isBf16()) 3 else 2) else (if (shader.isBf16()) 2 else 1);
        if (shader.isNvCoop2() and !shader.isBf16()) extensions[1] = zvk.extensions.nv_cooperative_matrix_2.name;
        device_info.pp_enabled_extension_names = extensions[0..device_info.enabled_extension_count].ptr;
    }

    var coop_features: zvk.PhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(zvk.PhysicalDeviceCooperativeMatrixFeaturesKHR);
    var nvcoop2_features: zvk.PhysicalDeviceCooperativeMatrix2FeaturesNV = std.mem.zeroes(zvk.PhysicalDeviceCooperativeMatrix2FeaturesNV);
    var storage16_features: zvk.PhysicalDevice16BitStorageFeatures = std.mem.zeroes(zvk.PhysicalDevice16BitStorageFeatures);
    var f16_features: zvk.PhysicalDeviceShaderFloat16Int8Features = std.mem.zeroes(zvk.PhysicalDeviceShaderFloat16Int8Features);
    var bf16_features: zvk.PhysicalDeviceShaderBfloat16FeaturesKHR = std.mem.zeroes(zvk.PhysicalDeviceShaderBfloat16FeaturesKHR);

    if (shader.isCoop()) {
        coop_features.s_type = .physical_device_cooperative_matrix_features_khr;
        coop_features.cooperative_matrix = zvk.Bool32.true;
        if (shader.isNvCoop2()) {
            nvcoop2_features.s_type = .physical_device_cooperative_matrix_2_features_nv;
            nvcoop2_features.cooperative_matrix_workgroup_scope = zvk.Bool32.true;
            nvcoop2_features.cooperative_matrix_flexible_dimensions = zvk.Bool32.true;
            nvcoop2_features.cooperative_matrix_tensor_addressing = zvk.Bool32.true;
            nvcoop2_features.cooperative_matrix_block_loads = zvk.Bool32.true;
            coop_features.p_next = &nvcoop2_features;
        }

        storage16_features.s_type = .physical_device_16bit_storage_features;
        storage16_features.storage_buffer_16_bit_access = zvk.Bool32.true;
        storage16_features.p_next = &coop_features;

        if (shader.isF16()) {
            f16_features.s_type = .physical_device_shader_float16_int8_features;
            f16_features.shader_float_16 = zvk.Bool32.true;
            f16_features.p_next = &storage16_features;
            device_info.p_next = &f16_features;
        } else {
            bf16_features.s_type = .physical_device_shader_bfloat16_features_khr;
            bf16_features.shader_b_float_16_type = zvk.Bool32.true;
            bf16_features.shader_b_float_16_cooperative_matrix = zvk.Bool32.true;
            bf16_features.p_next = &storage16_features;
            device_info.p_next = &bf16_features;
        }
    }

    var handle: zvk.Device = .null_handle;
    const create_result = vk.instance.dispatch.vkCreateDevice.?(physical_device, &device_info, null, &handle);
    if (create_result == zvk.Result.error_extension_not_present) return error.RequiredDeviceExtensionMissing;
    if (create_result == zvk.Result.error_feature_not_present) return error.RequiredDeviceFeatureMissing;
    try vkCheck(create_result);

    const fns = zvk.DeviceWrapper.load(handle, vk.instance.dispatch.vkGetDeviceProcAddr.?);

    var queue: zvk.Queue = .null_handle;
    fns.dispatch.vkGetDeviceQueue.?(handle, queue_family, 0, &queue);

    return .{
        .handle = handle,
        .fns = fns,
        .queue = queue,
        .physical_device = physical_device,
        .queue_family = queue_family,
    };
}

fn runMatmul(vk: *Vulkan, device: *Device, opts: Options) !void {
    if (opts.shader == .zig) {
        try runScalarMatmul(vk, device, opts);
    } else {
        try runCoopMatmul(vk, device, opts);
    }
}

fn runScalarMatmul(vk: *Vulkan, device: *Device, opts: Options) !void {
    const a_len = opts.m * opts.k;
    const b_len = opts.k * opts.n;
    const c_len = opts.m * opts.n;

    var a = try createBuffer(vk, device, a_len * @sizeOf(f32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(device, a);
    var b = try createBuffer(vk, device, b_len * @sizeOf(f32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(device, b);
    var out = try createBuffer(vk, device, c_len * @sizeOf(f32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(device, out);

    const a_values = a.slice(f32);
    const b_values = b.slice(f32);
    const out_values = out.slice(f32);
    fillInputA(a_values, opts.m, opts.k);
    fillInputB(b_values, opts.k, opts.n);
    @memset(out_values, 0);

    const shader_module = try createShaderModule(device, opts.shader);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);

    const descriptor_layout = try createDescriptorSetLayout(device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);

    const pipeline_layout = try createPipelineLayout(device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);

    const pipeline = try createPipeline(device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);

    const descriptor_pool = try createDescriptorPool(device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);

    const descriptor_set = try allocateDescriptorSet(device, descriptor_pool, descriptor_layout);
    updateDescriptorSet(device, descriptor_set, a, b, out);

    const command_pool = try createCommandPool(device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);

    const command_buffer = try allocateCommandBuffer(device, command_pool);
    try recordCommands(device, command_buffer, pipeline, pipeline_layout, descriptor_set, opts);

    const timed_cmd = try allocateCommandBuffer(device, command_pool);
    var query_pool: zvk.QueryPool = .null_handle;
    if (opts.timing == .gpu_timestamp) query_pool = try createTimestampQueryPool(device);
    defer if (query_pool != .null_handle) device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);
    if (opts.timing == .gpu_timestamp) try recordTimedCommands(device, timed_cmd, pipeline, pipeline_layout, descriptor_set, opts, query_pool);

    const fence = try createFence(device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);

    const timing = try timeDispatches(vk, device, command_buffer, timed_cmd, query_pool, fence, opts);

    if (!opts.skip_validation) try validate(out_values, a_values, b_values, opts.m, opts.n, opts.k);
    printResult(opts, timing);
}

fn runCoopMatmul(vk: *Vulkan, device: *Device, opts: Options) !void {
    const a_len = opts.m * opts.k;
    const b_len = opts.k * opts.n;
    const c_len = opts.m * opts.n;

    var a_stage = try createBuffer(vk, device, a_len * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(device, a_stage);
    var b_stage = try createBuffer(vk, device, b_len * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(device, b_stage);
    var out_stage = try createBuffer(vk, device, c_len * @sizeOf(f32), .{ .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(device, out_stage);

    const a_dev = try createBuffer(vk, device, a_stage.bytes(), .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(device, a_dev);
    const b_dev = try createBuffer(vk, device, b_stage.bytes(), .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(device, b_dev);
    const out_dev = try createBuffer(vk, device, out_stage.bytes(), .{ .storage_buffer_bit = true, .transfer_src_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(device, out_dev);

    const a_bits = a_stage.slice(u16);
    const b_bits = b_stage.slice(u16);
    const out_values = out_stage.slice(f32);
    fillInputAEncoded(a_bits, opts.m, opts.k, opts.shader);
    fillInputBEncoded(b_bits, opts.k, opts.n, opts.shader);
    @memset(out_values, 0);

    const shader_module = try createShaderModule(device, opts.shader);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);

    const descriptor_layout = try createDescriptorSetLayout(device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);

    const pipeline_layout = try createPipelineLayout(device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);

    const pipeline = try createPipeline(device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);

    const descriptor_pool = try createDescriptorPool(device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);

    const descriptor_set = try allocateDescriptorSet(device, descriptor_pool, descriptor_layout);
    updateDescriptorSet(device, descriptor_set, a_dev, b_dev, out_dev);

    const command_pool = try createCommandPool(device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);

    const upload_cmd = try allocateCommandBuffer(device, command_pool);
    const compute_cmd = try allocateCommandBuffer(device, command_pool);
    const timed_cmd = try allocateCommandBuffer(device, command_pool);
    const download_cmd = try allocateCommandBuffer(device, command_pool);
    try recordUploadCommands(device, upload_cmd, a_stage, a_dev, b_stage, b_dev);
    try recordCommands(device, compute_cmd, pipeline, pipeline_layout, descriptor_set, opts);
    var query_pool: zvk.QueryPool = .null_handle;
    if (opts.timing == .gpu_timestamp) query_pool = try createTimestampQueryPool(device);
    defer if (query_pool != .null_handle) device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);
    if (opts.timing == .gpu_timestamp) try recordTimedCommands(device, timed_cmd, pipeline, pipeline_layout, descriptor_set, opts, query_pool);
    if (!opts.skip_validation) try recordDownloadCommands(device, download_cmd, out_dev, out_stage);

    const fence = try createFence(device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);

    try submitCommand(device, upload_cmd, fence);
    const timing = try timeDispatches(vk, device, compute_cmd, timed_cmd, query_pool, fence, opts);
    if (!opts.skip_validation) try submitCommand(device, download_cmd, fence);

    if (!opts.skip_validation) try validateEncoded(out_values, opts);
    printResult(opts, timing);
}

fn createBuffer(vk: *Vulkan, device: *Device, byte_len: usize, usage: zvk.BufferUsageFlags, required: zvk.MemoryPropertyFlags, map: bool) !Buffer {
    var info: zvk.BufferCreateInfo = std.mem.zeroes(zvk.BufferCreateInfo);
    info.s_type = .buffer_create_info;
    info.size = byte_len;
    info.usage = usage;
    info.sharing_mode = zvk.SharingMode.exclusive;

    var handle: zvk.Buffer = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateBuffer.?(device.handle, &info, null, &handle));
    errdefer device.fns.dispatch.vkDestroyBuffer.?(device.handle, handle, null);

    var reqs: zvk.MemoryRequirements = undefined;
    device.fns.dispatch.vkGetBufferMemoryRequirements.?(device.handle, handle, &reqs);

    const memory_type = try findMemoryType(vk, device.physical_device, reqs.memory_type_bits, required);

    var alloc: zvk.MemoryAllocateInfo = std.mem.zeroes(zvk.MemoryAllocateInfo);
    alloc.s_type = .memory_allocate_info;
    alloc.allocation_size = reqs.size;
    alloc.memory_type_index = memory_type;

    var memory: zvk.DeviceMemory = .null_handle;
    try vkCheck(device.fns.dispatch.vkAllocateMemory.?(device.handle, &alloc, null, &memory));
    errdefer device.fns.dispatch.vkFreeMemory.?(device.handle, memory, null);

    try vkCheck(device.fns.dispatch.vkBindBufferMemory.?(device.handle, handle, memory, 0));

    var mapped: ?[*]u8 = null;
    if (map) {
        var mapped_raw: ?*anyopaque = null;
        try vkCheck(device.fns.dispatch.vkMapMemory.?(device.handle, memory, 0, alloc.allocation_size, .{}, &mapped_raw));
        mapped = @ptrCast(mapped_raw.?);
    }

    return .{ .handle = handle, .memory = memory, .mapped = mapped, .byte_len = byte_len };
}

fn destroyBuffer(device: *Device, buffer: Buffer) void {
    if (buffer.mapped != null) device.fns.dispatch.vkUnmapMemory.?(device.handle, buffer.memory);
    device.fns.dispatch.vkDestroyBuffer.?(device.handle, buffer.handle, null);
    device.fns.dispatch.vkFreeMemory.?(device.handle, buffer.memory, null);
}

fn findMemoryType(vk: *Vulkan, physical_device: zvk.PhysicalDevice, type_bits: u32, required: zvk.MemoryPropertyFlags) !u32 {
    var props: zvk.PhysicalDeviceMemoryProperties = undefined;
    vk.instance.dispatch.vkGetPhysicalDeviceMemoryProperties.?(physical_device, &props);

    for (props.memory_types[0..props.memory_type_count], 0..) |mem_type, i| {
        const bit: u32 = @as(u32, 1) << @intCast(i);
        if ((type_bits & bit) != 0 and mem_type.property_flags.contains(required)) return @intCast(i);
    }
    return error.MemoryTypeNotFound;
}

fn createShaderModule(device: *Device, shader: ShaderMode) !zvk.ShaderModule {
    const words = switch (shader) {
        .zig => matmul_zig_spv.words[0..],
        .coop_bf16 => matmul_coop_bf16_spv.words[0..],
        .coop_f16 => matmul_coop_f16_spv.words[0..],
        .coop_bf16_opt => matmul_coop_bf16_opt_spv.words[0..],
        .coop_f16_opt => matmul_coop_f16_opt_spv.words[0..],
        .coop_shared_f16 => matmul_coop_shared_f16_spv.words[0..],
        .nvcoop2_bf16 => matmul_nvcoop2_bf16_spv.words[0..],
        .nvcoop2_f16 => matmul_nvcoop2_f16_spv.words[0..],
        .nvcoop2_wide_f16 => matmul_nvcoop2_wide_f16_spv.words[0..],
        .nvcoop2_square_f16 => matmul_nvcoop2_square_f16_spv.words[0..],
        .nvcoop2_square_f16_frontend => matmul_nvcoop2_square_f16_frontend_spv.words[0..],
    };

    var info: zvk.ShaderModuleCreateInfo = std.mem.zeroes(zvk.ShaderModuleCreateInfo);
    info.s_type = .shader_module_create_info;
    info.code_size = words.len * @sizeOf(u32);
    info.p_code = words.ptr;

    var module: zvk.ShaderModule = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateShaderModule.?(device.handle, &info, null, &module));
    return module;
}

fn createDescriptorSetLayout(device: *Device) !zvk.DescriptorSetLayout {
    var bindings: [3]zvk.DescriptorSetLayoutBinding = undefined;
    for (&bindings, 0..) |*binding, i| {
        binding.* = std.mem.zeroes(zvk.DescriptorSetLayoutBinding);
        binding.binding = @intCast(i);
        binding.descriptor_type = zvk.DescriptorType.storage_buffer;
        binding.descriptor_count = 1;
        binding.stage_flags = zvk.ShaderStageFlags{ .compute_bit = true };
    }

    var info: zvk.DescriptorSetLayoutCreateInfo = std.mem.zeroes(zvk.DescriptorSetLayoutCreateInfo);
    info.s_type = .descriptor_set_layout_create_info;
    info.binding_count = bindings.len;
    info.p_bindings = bindings[0..].ptr;

    var layout: zvk.DescriptorSetLayout = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateDescriptorSetLayout.?(device.handle, &info, null, &layout));
    return layout;
}

fn createPipelineLayout(device: *Device, descriptor_layout: zvk.DescriptorSetLayout) !zvk.PipelineLayout {
    var push: zvk.PushConstantRange = std.mem.zeroes(zvk.PushConstantRange);
    push.stage_flags = zvk.ShaderStageFlags{ .compute_bit = true };
    push.offset = 0;
    push.size = @sizeOf(PushConstants);

    var info: zvk.PipelineLayoutCreateInfo = std.mem.zeroes(zvk.PipelineLayoutCreateInfo);
    info.s_type = .pipeline_layout_create_info;
    var layouts = [_]zvk.DescriptorSetLayout{descriptor_layout};
    var ranges = [_]zvk.PushConstantRange{push};
    info.set_layout_count = 1;
    info.p_set_layouts = layouts[0..].ptr;
    info.push_constant_range_count = 1;
    info.p_push_constant_ranges = ranges[0..].ptr;

    var layout: zvk.PipelineLayout = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreatePipelineLayout.?(device.handle, &info, null, &layout));
    return layout;
}

fn createPipeline(device: *Device, pipeline_layout: zvk.PipelineLayout, shader_module: zvk.ShaderModule) !zvk.Pipeline {
    var stage: zvk.PipelineShaderStageCreateInfo = std.mem.zeroes(zvk.PipelineShaderStageCreateInfo);
    stage.s_type = .pipeline_shader_stage_create_info;
    stage.stage = zvk.ShaderStageFlags{ .compute_bit = true };
    stage.module = shader_module;
    stage.p_name = "main";

    var info: zvk.ComputePipelineCreateInfo = std.mem.zeroes(zvk.ComputePipelineCreateInfo);
    info.s_type = .compute_pipeline_create_info;
    info.stage = stage;
    info.layout = pipeline_layout;

    var infos = [_]zvk.ComputePipelineCreateInfo{info};
    var pipelines = [_]zvk.Pipeline{.null_handle};
    try vkCheck(device.fns.dispatch.vkCreateComputePipelines.?(device.handle, .null_handle, 1, infos[0..].ptr, null, pipelines[0..].ptr));
    return pipelines[0];
}

fn createDescriptorPool(device: *Device) !zvk.DescriptorPool {
    var size: zvk.DescriptorPoolSize = std.mem.zeroes(zvk.DescriptorPoolSize);
    size.type = zvk.DescriptorType.storage_buffer;
    size.descriptor_count = 3;

    var info: zvk.DescriptorPoolCreateInfo = std.mem.zeroes(zvk.DescriptorPoolCreateInfo);
    info.s_type = .descriptor_pool_create_info;
    info.max_sets = 1;
    info.pool_size_count = 1;
    var sizes = [_]zvk.DescriptorPoolSize{size};
    info.p_pool_sizes = sizes[0..].ptr;

    var pool: zvk.DescriptorPool = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateDescriptorPool.?(device.handle, &info, null, &pool));
    return pool;
}

fn allocateDescriptorSet(device: *Device, pool: zvk.DescriptorPool, layout: zvk.DescriptorSetLayout) !zvk.DescriptorSet {
    var info: zvk.DescriptorSetAllocateInfo = std.mem.zeroes(zvk.DescriptorSetAllocateInfo);
    info.s_type = .descriptor_set_allocate_info;
    info.descriptor_pool = pool;
    info.descriptor_set_count = 1;
    var layouts = [_]zvk.DescriptorSetLayout{layout};
    info.p_set_layouts = layouts[0..].ptr;

    var sets = [_]zvk.DescriptorSet{.null_handle};
    try vkCheck(device.fns.dispatch.vkAllocateDescriptorSets.?(device.handle, &info, sets[0..].ptr));
    return sets[0];
}

fn updateDescriptorSet(device: *Device, set: zvk.DescriptorSet, a: Buffer, b: Buffer, out: Buffer) void {
    var infos = [_]zvk.DescriptorBufferInfo{
        .{ .buffer = a.handle, .offset = 0, .range = a.bytes() },
        .{ .buffer = b.handle, .offset = 0, .range = b.bytes() },
        .{ .buffer = out.handle, .offset = 0, .range = out.bytes() },
    };
    var writes: [3]zvk.WriteDescriptorSet = undefined;
    for (&writes, 0..) |*write, i| {
        write.* = std.mem.zeroes(zvk.WriteDescriptorSet);
        write.s_type = .write_descriptor_set;
        write.dst_set = set;
        write.dst_binding = @intCast(i);
        write.descriptor_count = 1;
        write.descriptor_type = zvk.DescriptorType.storage_buffer;
        write.p_buffer_info = @ptrCast(&infos[i]);
    }
    device.fns.dispatch.vkUpdateDescriptorSets.?(device.handle, writes.len, &writes, 0, null);
}

fn createCommandPool(device: *Device) !zvk.CommandPool {
    var info: zvk.CommandPoolCreateInfo = std.mem.zeroes(zvk.CommandPoolCreateInfo);
    info.s_type = .command_pool_create_info;
    info.queue_family_index = device.queue_family;

    var pool: zvk.CommandPool = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateCommandPool.?(device.handle, &info, null, &pool));
    return pool;
}

fn allocateCommandBuffer(device: *Device, pool: zvk.CommandPool) !zvk.CommandBuffer {
    var info: zvk.CommandBufferAllocateInfo = std.mem.zeroes(zvk.CommandBufferAllocateInfo);
    info.s_type = .command_buffer_allocate_info;
    info.command_pool = pool;
    info.level = zvk.CommandBufferLevel.primary;
    info.command_buffer_count = 1;

    var buffers = [_]zvk.CommandBuffer{.null_handle};
    try vkCheck(device.fns.dispatch.vkAllocateCommandBuffers.?(device.handle, &info, buffers[0..].ptr));
    return buffers[0];
}

fn recordCommands(device: *Device, command_buffer: zvk.CommandBuffer, pipeline: zvk.Pipeline, pipeline_layout: zvk.PipelineLayout, descriptor_set: zvk.DescriptorSet, opts: Options) !void {
    return recordRepeatedCommands(device, command_buffer, pipeline, pipeline_layout, descriptor_set, opts, 1, .null_handle);
}

fn recordTimedCommands(device: *Device, command_buffer: zvk.CommandBuffer, pipeline: zvk.Pipeline, pipeline_layout: zvk.PipelineLayout, descriptor_set: zvk.DescriptorSet, opts: Options, query_pool: zvk.QueryPool) !void {
    return recordRepeatedCommands(device, command_buffer, pipeline, pipeline_layout, descriptor_set, opts, opts.iters, query_pool);
}

fn recordRepeatedCommands(device: *Device, command_buffer: zvk.CommandBuffer, pipeline: zvk.Pipeline, pipeline_layout: zvk.PipelineLayout, descriptor_set: zvk.DescriptorSet, opts: Options, repeat_count: usize, query_pool: zvk.QueryPool) !void {
    var begin: zvk.CommandBufferBeginInfo = std.mem.zeroes(zvk.CommandBufferBeginInfo);
    begin.s_type = .command_buffer_begin_info;
    try vkCheck(device.fns.dispatch.vkBeginCommandBuffer.?(command_buffer, &begin));

    if (query_pool != .null_handle) {
        device.fns.dispatch.vkCmdResetQueryPool.?(command_buffer, query_pool, 0, 2);
        device.fns.dispatch.vkCmdWriteTimestamp.?(command_buffer, zvk.PipelineStageFlags{ .top_of_pipe_bit = true }, query_pool, 0);
    }

    const pc: PushConstants = .{
        .m = @intCast(opts.m),
        .n = @intCast(opts.n),
        .k = @intCast(opts.k),
        .a_stride = @intCast(opts.k),
        .b_stride = @intCast(opts.n),
        .c_stride = @intCast(opts.n),
    };

    device.fns.dispatch.vkCmdBindPipeline.?(command_buffer, zvk.PipelineBindPoint.compute, pipeline);
    var sets = [_]zvk.DescriptorSet{descriptor_set};
    device.fns.dispatch.vkCmdBindDescriptorSets.?(command_buffer, zvk.PipelineBindPoint.compute, pipeline_layout, 0, 1, sets[0..].ptr, 0, null);
    device.fns.dispatch.vkCmdPushConstants.?(command_buffer, pipeline_layout, zvk.ShaderStageFlags{ .compute_bit = true }, 0, @sizeOf(PushConstants), &pc);
    const dispatch_x: u32 = if (opts.shader.isCoop()) @intCast(opts.n / opts.shader.outputTileN()) else roundUpDiv(@intCast(opts.n), 16);
    const dispatch_y: u32 = if (opts.shader.isCoop()) @intCast(opts.m / opts.shader.outputTileM()) else roundUpDiv(@intCast(opts.m), 16);
    for (0..repeat_count) |_| device.fns.dispatch.vkCmdDispatch.?(command_buffer, dispatch_x, dispatch_y, 1);

    if (query_pool != .null_handle) {
        device.fns.dispatch.vkCmdWriteTimestamp.?(command_buffer, zvk.PipelineStageFlags{ .bottom_of_pipe_bit = true }, query_pool, 1);
    }

    try vkCheck(device.fns.dispatch.vkEndCommandBuffer.?(command_buffer));
}

fn recordUploadCommands(device: *Device, command_buffer: zvk.CommandBuffer, a_stage: Buffer, a_dev: Buffer, b_stage: Buffer, b_dev: Buffer) !void {
    var begin: zvk.CommandBufferBeginInfo = std.mem.zeroes(zvk.CommandBufferBeginInfo);
    begin.s_type = .command_buffer_begin_info;
    try vkCheck(device.fns.dispatch.vkBeginCommandBuffer.?(command_buffer, &begin));

    copyBuffer(device, command_buffer, a_stage, a_dev);
    copyBuffer(device, command_buffer, b_stage, b_dev);

    var barriers = [_]zvk.BufferMemoryBarrier{
        bufferBarrier(a_dev, zvk.AccessFlags{ .transfer_write_bit = true }, zvk.AccessFlags{ .shader_read_bit = true }),
        bufferBarrier(b_dev, zvk.AccessFlags{ .transfer_write_bit = true }, zvk.AccessFlags{ .shader_read_bit = true }),
    };
    device.fns.dispatch.vkCmdPipelineBarrier.?(command_buffer, zvk.PipelineStageFlags{ .transfer_bit = true }, zvk.PipelineStageFlags{ .compute_shader_bit = true }, .{}, 0, null, barriers.len, barriers[0..].ptr, 0, null);

    try vkCheck(device.fns.dispatch.vkEndCommandBuffer.?(command_buffer));
}

fn recordDownloadCommands(device: *Device, command_buffer: zvk.CommandBuffer, out_dev: Buffer, out_stage: Buffer) !void {
    var begin: zvk.CommandBufferBeginInfo = std.mem.zeroes(zvk.CommandBufferBeginInfo);
    begin.s_type = .command_buffer_begin_info;
    try vkCheck(device.fns.dispatch.vkBeginCommandBuffer.?(command_buffer, &begin));

    var barriers = [_]zvk.BufferMemoryBarrier{
        bufferBarrier(out_dev, zvk.AccessFlags{ .shader_write_bit = true }, zvk.AccessFlags{ .transfer_read_bit = true }),
    };
    device.fns.dispatch.vkCmdPipelineBarrier.?(command_buffer, zvk.PipelineStageFlags{ .compute_shader_bit = true }, zvk.PipelineStageFlags{ .transfer_bit = true }, .{}, 0, null, barriers.len, barriers[0..].ptr, 0, null);
    copyBuffer(device, command_buffer, out_dev, out_stage);

    try vkCheck(device.fns.dispatch.vkEndCommandBuffer.?(command_buffer));
}

fn copyBuffer(device: *Device, command_buffer: zvk.CommandBuffer, src: Buffer, dst: Buffer) void {
    var regions = [_]zvk.BufferCopy{
        .{ .src_offset = 0, .dst_offset = 0, .size = src.bytes() },
    };
    device.fns.dispatch.vkCmdCopyBuffer.?(command_buffer, src.handle, dst.handle, 1, regions[0..].ptr);
}

fn bufferBarrier(buffer: Buffer, src_access: zvk.AccessFlags, dst_access: zvk.AccessFlags) zvk.BufferMemoryBarrier {
    return .{
        .s_type = .buffer_memory_barrier,
        .p_next = null,
        .src_access_mask = src_access,
        .dst_access_mask = dst_access,
        .src_queue_family_index = zvk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = zvk.QUEUE_FAMILY_IGNORED,
        .buffer = buffer.handle,
        .offset = 0,
        .size = buffer.bytes(),
    };
}

fn createFence(device: *Device) !zvk.Fence {
    var info: zvk.FenceCreateInfo = std.mem.zeroes(zvk.FenceCreateInfo);
    info.s_type = .fence_create_info;

    var fence: zvk.Fence = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateFence.?(device.handle, &info, null, &fence));
    return fence;
}

fn createTimestampQueryPool(device: *Device) !zvk.QueryPool {
    var info: zvk.QueryPoolCreateInfo = std.mem.zeroes(zvk.QueryPoolCreateInfo);
    info.s_type = .query_pool_create_info;
    info.query_type = zvk.QueryType.timestamp;
    info.query_count = 2;

    var pool: zvk.QueryPool = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateQueryPool.?(device.handle, &info, null, &pool));
    return pool;
}

fn timeDispatches(vk: *Vulkan, device: *Device, command_buffer: zvk.CommandBuffer, timed_command_buffer: zvk.CommandBuffer, query_pool: zvk.QueryPool, fence: zvk.Fence, opts: Options) !TimingResult {
    if (opts.timing == .submit_cpu) {
        const avg_ns = try timeRepeatedDispatch(device, command_buffer, fence, opts.warmup, opts.iters);
        return .{ .avg_ns = avg_ns };
    }

    try requireTimestampQueue(vk, device);
    for (0..opts.warmup) |_| try submitCommand(device, command_buffer, fence);

    const start = try nanoTimestamp();
    try submitCommand(device, timed_command_buffer, fence);
    const end = try nanoTimestamp();

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(
        device.handle,
        query_pool,
        0,
        2,
        @sizeOf(@TypeOf(timestamps)),
        &timestamps,
        @sizeOf(u64),
        zvk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true },
    ));

    const elapsed_ticks = timestamps[1] - timestamps[0];
    const gpu_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(vk, device.physical_device)) / @as(f64, @floatFromInt(opts.iters));
    const batch_cpu_ns = @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(opts.iters));
    return .{ .avg_ns = gpu_ns, .avg_gpu_ns = gpu_ns, .avg_batch_cpu_ns = batch_cpu_ns };
}

fn timeRepeatedDispatch(device: *Device, command_buffer: zvk.CommandBuffer, fence: zvk.Fence, warmup: usize, iters: usize) !f64 {
    for (0..warmup) |_| try submitCommand(device, command_buffer, fence);

    const start = try nanoTimestamp();
    for (0..iters) |_| try submitCommand(device, command_buffer, fence);
    const end = try nanoTimestamp();

    const elapsed: f64 = @floatFromInt(end - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn requireTimestampQueue(vk: *Vulkan, device: *Device) !void {
    var count: u32 = 0;
    vk.instance.dispatch.vkGetPhysicalDeviceQueueFamilyProperties.?(device.physical_device, &count, null);
    if (device.queue_family >= count) return error.TimestampUnsupported;

    var stack_families: [64]zvk.QueueFamilyProperties = undefined;
    if (count > stack_families.len) return error.TimestampUnsupported;
    vk.instance.dispatch.vkGetPhysicalDeviceQueueFamilyProperties.?(device.physical_device, &count, stack_families[0..].ptr);
    if (stack_families[device.queue_family].timestamp_valid_bits == 0) {
        log.err("selected compute queue family does not support timestamps; use --timing=submit-cpu", .{});
        return error.TimestampUnsupported;
    }
}

fn nanoTimestamp() !i128 {
    var ts: Timespec = undefined;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return error.ClockFailure;
    return @as(i128, ts.tv_sec) * 1_000_000_000 + ts.tv_nsec;
}

fn submitCommand(device: *Device, command_buffer: zvk.CommandBuffer, fence: zvk.Fence) !void {
    var submit: zvk.SubmitInfo = std.mem.zeroes(zvk.SubmitInfo);
    submit.s_type = .submit_info;
    submit.command_buffer_count = 1;
    var command_buffers = [_]zvk.CommandBuffer{command_buffer};
    submit.p_command_buffers = command_buffers[0..].ptr;
    var submits = [_]zvk.SubmitInfo{submit};
    try vkCheck(device.fns.dispatch.vkQueueSubmit.?(device.queue, 1, submits[0..].ptr, fence));
    var fences = [_]zvk.Fence{fence};
    try vkCheck(device.fns.dispatch.vkWaitForFences.?(device.handle, 1, fences[0..].ptr, zvk.Bool32.true, std.math.maxInt(u64)));
    try vkCheck(device.fns.dispatch.vkResetFences.?(device.handle, 1, fences[0..].ptr));
}

fn printResult(opts: Options, timing: TimingResult) void {
    const ops: f64 = 2.0 * @as(f64, @floatFromInt(opts.m)) * @as(f64, @floatFromInt(opts.n)) * @as(f64, @floatFromInt(opts.k));
    const tflops = ops / timing.avg_ns / 1.0e3;
    const status = if (opts.skip_validation) "benchmark complete" else "validation passed";
    if (timing.avg_gpu_ns) |avg_gpu_ns| {
        std.debug.print("{s}: shader={s} m={d} n={d} k={d} avg_gpu_ns={d:.2} avg_batch_cpu_ns={d:.2} TFLOP/s={d:.4}\n", .{
            status,
            opts.shader.name(),
            opts.m,
            opts.n,
            opts.k,
            avg_gpu_ns,
            timing.avg_batch_cpu_ns.?,
            tflops,
        });
    } else {
        std.debug.print("{s}: shader={s} m={d} n={d} k={d} avg_ns={d:.2} TFLOP/s={d:.4}\n", .{
            status,
            opts.shader.name(),
            opts.m,
            opts.n,
            opts.k,
            timing.avg_ns,
            tflops,
        });
    }
}

fn fillInputA(dst: []f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        for (0..cols) |col| {
            const raw: f32 = @floatFromInt((r * 17 + col * 13 + 3) % 29);
            dst[r * cols + col] = (raw - 14.0) / 7.0;
        }
    }
}

fn fillInputB(dst: []f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        for (0..cols) |col| {
            const raw: f32 = @floatFromInt((r * 11 + col * 5 + 7) % 31);
            dst[r * cols + col] = (raw - 15.0) / 9.0;
        }
    }
}

fn fillInputAEncoded(dst: []u16, rows: usize, cols: usize, shader: ShaderMode) void {
    for (0..rows) |r| {
        for (0..cols) |col| {
            const raw: f32 = @floatFromInt((r * 17 + col * 13 + 3) % 29);
            dst[r * cols + col] = encodeInput((raw - 14.0) / 7.0, shader);
        }
    }
}

fn fillInputBEncoded(dst: []u16, rows: usize, cols: usize, shader: ShaderMode) void {
    for (0..rows) |r| {
        for (0..cols) |col| {
            const raw: f32 = @floatFromInt((r * 11 + col * 5 + 7) % 31);
            dst[r * cols + col] = encodeInput((raw - 15.0) / 9.0, shader);
        }
    }
}

fn validate(got: []const f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) !void {
    for (0..m) |row| {
        for (0..n) |col| {
            var expected: f32 = 0;
            for (0..k) |i| expected += a[row * k + i] * b[i * n + col];
            const actual = got[row * n + col];
            if (@abs(actual - expected) > 1e-4) {
                log.err("mismatch at ({d}, {d}): got {d:.6}, expected {d:.6}", .{ row, col, actual, expected });
                return error.ValidationFailed;
            }
        }
    }
}

fn validateEncoded(got: []const f32, opts: Options) !void {
    const abs_tol: f32 = if (opts.shader.isF16()) 1e-2 else 2e-1;
    const rel_tol: f32 = if (opts.shader.isF16()) 5e-3 else 2e-2;
    var expected_by_residue: [29][31]f32 = undefined;
    for (0..29) |row_residue| {
        for (0..31) |col_residue| {
            var expected: f32 = 0;
            for (0..opts.k) |i| {
                expected += inputAQuantized(row_residue, i, opts.shader) * inputBQuantized(i, col_residue, opts.shader);
            }
            expected_by_residue[row_residue][col_residue] = expected;
        }
    }

    for (0..opts.m) |row| {
        for (0..opts.n) |col| {
            const expected = expected_by_residue[row % 29][col % 31];
            const actual = got[row * opts.n + col];
            const tolerance = @max(abs_tol, @abs(expected) * rel_tol);
            if (@abs(actual - expected) > tolerance) {
                log.err("mismatch at ({d}, {d}): got {d:.6}, expected {d:.6}, tolerance {d:.6}", .{ row, col, actual, expected, tolerance });
                return error.ValidationFailed;
            }
        }
    }
}

fn inputAQuantized(row: usize, col: usize, shader: ShaderMode) f32 {
    const raw: f32 = @floatFromInt((row * 17 + col * 13 + 3) % 29);
    return decodeInput(encodeInput((raw - 14.0) / 7.0, shader), shader);
}

fn inputBQuantized(row: usize, col: usize, shader: ShaderMode) f32 {
    const raw: f32 = @floatFromInt((row * 11 + col * 5 + 7) % 31);
    return decodeInput(encodeInput((raw - 15.0) / 9.0, shader), shader);
}

fn encodeInput(value: f32, shader: ShaderMode) u16 {
    return switch (shader) {
        .coop_f16, .coop_f16_opt, .coop_shared_f16, .nvcoop2_f16, .nvcoop2_wide_f16, .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => f32ToF16Bits(value),
        .coop_bf16, .coop_bf16_opt, .nvcoop2_bf16 => f32ToBf16Bits(value),
        .zig => unreachable,
    };
}

fn decodeInput(bits: u16, shader: ShaderMode) f32 {
    return switch (shader) {
        .coop_f16, .coop_f16_opt, .coop_shared_f16, .nvcoop2_f16, .nvcoop2_wide_f16, .nvcoop2_square_f16, .nvcoop2_square_f16_frontend => f16BitsToF32(bits),
        .coop_bf16, .coop_bf16_opt, .nvcoop2_bf16 => bf16BitsToF32(bits),
        .zig => unreachable,
    };
}

fn f32ToF16Bits(value: f32) u16 {
    const half: f16 = @floatCast(value);
    return @bitCast(half);
}

fn f16BitsToF32(bits: u16) f32 {
    const half: f16 = @bitCast(bits);
    return @floatCast(half);
}

fn f32ToBf16Bits(value: f32) u16 {
    const raw: u32 = @bitCast(value);
    const rounded = raw + 0x7fff + ((raw >> 16) & 1);
    return @truncate(rounded >> 16);
}

fn bf16BitsToF32(bits: u16) f32 {
    const raw = @as(u32, bits) << 16;
    return @bitCast(raw);
}

fn roundUpDiv(value: u32, divisor: u32) u32 {
    return (value + divisor - 1) / divisor;
}

fn deviceName(props: *const zvk.PhysicalDeviceProperties) []const u8 {
    return std.mem.sliceTo(&props.device_name, 0);
}

fn api_version(props: *const zvk.PhysicalDeviceProperties) u32 {
    return props.api_version;
}

fn versionMajor(version: u32) u32 {
    return @as(zvk.Version, @bitCast(version)).major;
}

fn versionMinor(version: u32) u32 {
    return @as(zvk.Version, @bitCast(version)).minor;
}

fn versionPatch(version: u32) u32 {
    return @as(zvk.Version, @bitCast(version)).patch;
}

fn vendorId(props: *const zvk.PhysicalDeviceProperties) u32 {
    return props.vendor_id;
}

fn deviceType(props: *const zvk.PhysicalDeviceProperties) zvk.PhysicalDeviceType {
    return props.device_type;
}

fn timestampPeriodNs(vk: *Vulkan, physical_device: zvk.PhysicalDevice) f32 {
    var props: zvk.PhysicalDeviceProperties = undefined;
    vk.instance.dispatch.vkGetPhysicalDeviceProperties.?(physical_device, &props);
    return props.limits.timestamp_period;
}

fn vkCheck(result: zvk.Result) !void {
    if (result == zvk.Result.success) return;
    log.err("Vulkan error: {d}", .{@intFromEnum(result)});
    return error.VulkanFailure;
}
