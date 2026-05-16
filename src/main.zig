const std = @import("std");
const matmul_coop_bf16_spv = @import("matmul_coop_bf16_spv");
const matmul_coop_f16_spv = @import("matmul_coop_f16_spv");
const matmul_zig_spv = @import("matmul_zig_spv");
const c = @import("vulkan_loader.zig").c;

const log = std.log.scoped(.vk_matmul);

const VkGetInstanceProcAddr = *const fn (c.VkInstance, [*:0]const u8) callconv(.c) c.PFN_vkVoidFunction;
const VkGetDeviceProcAddr = *const fn (c.VkDevice, [*:0]const u8) callconv(.c) c.PFN_vkVoidFunction;

const CLOCK_MONOTONIC: c_int = 1;
const Timespec = extern struct {
    tv_sec: isize,
    tv_nsec: isize,
};
extern fn clock_gettime(clk_id: c_int, tp: *Timespec) callconv(.c) c_int;

const VkCreateInstance = *const fn (*const c.VkInstanceCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkInstance) callconv(.c) c.VkResult;
const VkDestroyInstance = *const fn (c.VkInstance, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkEnumeratePhysicalDevices = *const fn (c.VkInstance, *u32, ?[*]c.VkPhysicalDevice) callconv(.c) c.VkResult;
const VkGetPhysicalDeviceProperties = *const fn (c.VkPhysicalDevice, *c.VkPhysicalDeviceProperties) callconv(.c) void;
const VkGetPhysicalDeviceMemoryProperties = *const fn (c.VkPhysicalDevice, *c.VkPhysicalDeviceMemoryProperties) callconv(.c) void;
const VkGetPhysicalDeviceQueueFamilyProperties = *const fn (c.VkPhysicalDevice, *u32, ?[*]c.VkQueueFamilyProperties) callconv(.c) void;
const VkEnumerateDeviceExtensionProperties = *const fn (c.VkPhysicalDevice, ?[*:0]const u8, *u32, ?[*]c.VkExtensionProperties) callconv(.c) c.VkResult;
const VkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = *const fn (c.VkPhysicalDevice, *u32, ?[*]c.VkCooperativeMatrixPropertiesKHR) callconv(.c) c.VkResult;
const VkCreateDevice = *const fn (c.VkPhysicalDevice, *const c.VkDeviceCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkDevice) callconv(.c) c.VkResult;
const VkDestroyDevice = *const fn (c.VkDevice, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkGetDeviceQueue = *const fn (c.VkDevice, u32, u32, *c.VkQueue) callconv(.c) void;
const VkCreateBuffer = *const fn (c.VkDevice, *const c.VkBufferCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkBuffer) callconv(.c) c.VkResult;
const VkDestroyBuffer = *const fn (c.VkDevice, c.VkBuffer, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkGetBufferMemoryRequirements = *const fn (c.VkDevice, c.VkBuffer, *c.VkMemoryRequirements) callconv(.c) void;
const VkAllocateMemory = *const fn (c.VkDevice, *const c.VkMemoryAllocateInfo, ?*const c.VkAllocationCallbacks, *c.VkDeviceMemory) callconv(.c) c.VkResult;
const VkFreeMemory = *const fn (c.VkDevice, c.VkDeviceMemory, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkBindBufferMemory = *const fn (c.VkDevice, c.VkBuffer, c.VkDeviceMemory, c.VkDeviceSize) callconv(.c) c.VkResult;
const VkMapMemory = *const fn (c.VkDevice, c.VkDeviceMemory, c.VkDeviceSize, c.VkDeviceSize, c.VkMemoryMapFlags, *?*anyopaque) callconv(.c) c.VkResult;
const VkUnmapMemory = *const fn (c.VkDevice, c.VkDeviceMemory) callconv(.c) void;
const VkCreateShaderModule = *const fn (c.VkDevice, *const c.VkShaderModuleCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkShaderModule) callconv(.c) c.VkResult;
const VkDestroyShaderModule = *const fn (c.VkDevice, c.VkShaderModule, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkCreateDescriptorSetLayout = *const fn (c.VkDevice, *const c.VkDescriptorSetLayoutCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkDescriptorSetLayout) callconv(.c) c.VkResult;
const VkDestroyDescriptorSetLayout = *const fn (c.VkDevice, c.VkDescriptorSetLayout, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkCreatePipelineLayout = *const fn (c.VkDevice, *const c.VkPipelineLayoutCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkPipelineLayout) callconv(.c) c.VkResult;
const VkDestroyPipelineLayout = *const fn (c.VkDevice, c.VkPipelineLayout, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkCreateComputePipelines = *const fn (c.VkDevice, c.VkPipelineCache, u32, [*]const c.VkComputePipelineCreateInfo, ?*const c.VkAllocationCallbacks, [*]c.VkPipeline) callconv(.c) c.VkResult;
const VkDestroyPipeline = *const fn (c.VkDevice, c.VkPipeline, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkCreateDescriptorPool = *const fn (c.VkDevice, *const c.VkDescriptorPoolCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkDescriptorPool) callconv(.c) c.VkResult;
const VkDestroyDescriptorPool = *const fn (c.VkDevice, c.VkDescriptorPool, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkAllocateDescriptorSets = *const fn (c.VkDevice, *const c.VkDescriptorSetAllocateInfo, [*]c.VkDescriptorSet) callconv(.c) c.VkResult;
const VkUpdateDescriptorSets = *const fn (c.VkDevice, u32, [*]const c.VkWriteDescriptorSet, u32, ?*const anyopaque) callconv(.c) void;
const VkCreateCommandPool = *const fn (c.VkDevice, *const c.VkCommandPoolCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkCommandPool) callconv(.c) c.VkResult;
const VkDestroyCommandPool = *const fn (c.VkDevice, c.VkCommandPool, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkAllocateCommandBuffers = *const fn (c.VkDevice, *const c.VkCommandBufferAllocateInfo, [*]c.VkCommandBuffer) callconv(.c) c.VkResult;
const VkBeginCommandBuffer = *const fn (c.VkCommandBuffer, *const c.VkCommandBufferBeginInfo) callconv(.c) c.VkResult;
const VkEndCommandBuffer = *const fn (c.VkCommandBuffer) callconv(.c) c.VkResult;
const VkCmdBindPipeline = *const fn (c.VkCommandBuffer, c.VkPipelineBindPoint, c.VkPipeline) callconv(.c) void;
const VkCmdBindDescriptorSets = *const fn (c.VkCommandBuffer, c.VkPipelineBindPoint, c.VkPipelineLayout, u32, u32, [*]const c.VkDescriptorSet, u32, ?[*]const u32) callconv(.c) void;
const VkCmdPushConstants = *const fn (c.VkCommandBuffer, c.VkPipelineLayout, c.VkShaderStageFlags, u32, u32, *const anyopaque) callconv(.c) void;
const VkCmdCopyBuffer = *const fn (c.VkCommandBuffer, c.VkBuffer, c.VkBuffer, u32, [*]const c.VkBufferCopy) callconv(.c) void;
const VkCmdPipelineBarrier = *const fn (c.VkCommandBuffer, c.VkPipelineStageFlags, c.VkPipelineStageFlags, c.VkDependencyFlags, u32, ?*const anyopaque, u32, ?[*]const c.VkBufferMemoryBarrier, u32, ?*const anyopaque) callconv(.c) void;
const VkCmdDispatch = *const fn (c.VkCommandBuffer, u32, u32, u32) callconv(.c) void;
const VkCreateFence = *const fn (c.VkDevice, *const c.VkFenceCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkFence) callconv(.c) c.VkResult;
const VkDestroyFence = *const fn (c.VkDevice, c.VkFence, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkQueueSubmit = *const fn (c.VkQueue, u32, [*]const c.VkSubmitInfo, c.VkFence) callconv(.c) c.VkResult;
const VkWaitForFences = *const fn (c.VkDevice, u32, [*]const c.VkFence, c.VkBool32, u64) callconv(.c) c.VkResult;
const VkResetFences = *const fn (c.VkDevice, u32, [*]const c.VkFence) callconv(.c) c.VkResult;
const VkDeviceWaitIdle = *const fn (c.VkDevice) callconv(.c) c.VkResult;

const InstanceFns = struct {
    destroyInstance: VkDestroyInstance,
    enumeratePhysicalDevices: VkEnumeratePhysicalDevices,
    getPhysicalDeviceProperties: VkGetPhysicalDeviceProperties,
    getPhysicalDeviceMemoryProperties: VkGetPhysicalDeviceMemoryProperties,
    getPhysicalDeviceQueueFamilyProperties: VkGetPhysicalDeviceQueueFamilyProperties,
    enumerateDeviceExtensionProperties: VkEnumerateDeviceExtensionProperties,
    getCooperativeMatrixPropertiesKHR: ?VkGetPhysicalDeviceCooperativeMatrixPropertiesKHR,
    createDevice: VkCreateDevice,
    getDeviceProcAddr: VkGetDeviceProcAddr,
};

const DeviceFns = struct {
    destroyDevice: VkDestroyDevice,
    getDeviceQueue: VkGetDeviceQueue,
    createBuffer: VkCreateBuffer,
    destroyBuffer: VkDestroyBuffer,
    getBufferMemoryRequirements: VkGetBufferMemoryRequirements,
    allocateMemory: VkAllocateMemory,
    freeMemory: VkFreeMemory,
    bindBufferMemory: VkBindBufferMemory,
    mapMemory: VkMapMemory,
    unmapMemory: VkUnmapMemory,
    createShaderModule: VkCreateShaderModule,
    destroyShaderModule: VkDestroyShaderModule,
    createDescriptorSetLayout: VkCreateDescriptorSetLayout,
    destroyDescriptorSetLayout: VkDestroyDescriptorSetLayout,
    createPipelineLayout: VkCreatePipelineLayout,
    destroyPipelineLayout: VkDestroyPipelineLayout,
    createComputePipelines: VkCreateComputePipelines,
    destroyPipeline: VkDestroyPipeline,
    createDescriptorPool: VkCreateDescriptorPool,
    destroyDescriptorPool: VkDestroyDescriptorPool,
    allocateDescriptorSets: VkAllocateDescriptorSets,
    updateDescriptorSets: VkUpdateDescriptorSets,
    createCommandPool: VkCreateCommandPool,
    destroyCommandPool: VkDestroyCommandPool,
    allocateCommandBuffers: VkAllocateCommandBuffers,
    beginCommandBuffer: VkBeginCommandBuffer,
    endCommandBuffer: VkEndCommandBuffer,
    cmdBindPipeline: VkCmdBindPipeline,
    cmdBindDescriptorSets: VkCmdBindDescriptorSets,
    cmdPushConstants: VkCmdPushConstants,
    cmdCopyBuffer: VkCmdCopyBuffer,
    cmdPipelineBarrier: VkCmdPipelineBarrier,
    cmdDispatch: VkCmdDispatch,
    createFence: VkCreateFence,
    destroyFence: VkDestroyFence,
    queueSubmit: VkQueueSubmit,
    waitForFences: VkWaitForFences,
    resetFences: VkResetFences,
    deviceWaitIdle: VkDeviceWaitIdle,
};

const ShaderMode = enum {
    zig,
    coop_bf16,
    coop_f16,

    fn isCoop(self: ShaderMode) bool {
        return self != .zig;
    }

    fn name(self: ShaderMode) []const u8 {
        return switch (self) {
            .zig => "zig",
            .coop_bf16 => "coop",
            .coop_f16 => "coop-f16",
        };
    }

    fn inputComponentType(self: ShaderMode) c.VkComponentTypeKHR {
        return switch (self) {
            .zig => c.VK_COMPONENT_TYPE_FLOAT32_KHR,
            .coop_bf16 => c.VK_COMPONENT_TYPE_BFLOAT16_KHR,
            .coop_f16 => c.VK_COMPONENT_TYPE_FLOAT16_KHR,
        };
    }

    fn tileN(self: ShaderMode) usize {
        return switch (self) {
            .zig => 16,
            .coop_bf16 => 16,
            .coop_f16 => 8,
        };
    }
};

const Options = struct {
    shader: ShaderMode = .zig,
    m: usize = 64,
    n: usize = 64,
    k: usize = 64,
    iters: usize = 50,
    warmup: usize = 5,
    list_devices: bool = false,
    device_index: ?usize = null,
    device_substr: ?[]const u8 = null,
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
    handle: c.VkBuffer,
    memory: c.VkDeviceMemory,
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
    if (opts.shader.isCoop()) try validateCoopDimensions(opts);

    var vk = try Vulkan.open();
    defer vk.close();

    const instance = try vk.createInstance();
    defer vk.instance.destroyInstance(instance, null);

    const selected = try selectPhysicalDevice(&vk, allocator, instance, opts.device_index, opts.device_substr, opts.list_devices);
    if (opts.list_devices) return;

    if (opts.shader.isCoop()) try requireCoopMatrixProperty(&vk, allocator, selected.physical_device, opts.shader);

    var device = try createDevice(&vk, allocator, selected.physical_device, selected.queue_family, opts.shader);
    defer device.deinit();

    try runMatmul(&vk, &device, opts);
}

fn parseArgs(args: []const []const u8) !Options {
    var opts: Options = .{};
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--list-devices")) {
            opts.list_devices = true;
        } else if (std.mem.startsWith(u8, arg, "--shader=")) {
            const value = arg["--shader=".len..];
            if (std.mem.eql(u8, value, "zig")) {
                opts.shader = .zig;
            } else if (std.mem.eql(u8, value, "coop") or std.mem.eql(u8, value, "coop-bf16")) {
                opts.shader = .coop_bf16;
            } else if (std.mem.eql(u8, value, "coop-f16")) {
                opts.shader = .coop_f16;
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
    getInstanceProcAddr: VkGetInstanceProcAddr,
    createInstanceFn: VkCreateInstance,
    instance: InstanceFns,

    fn open() !Vulkan {
        var lib = try std.DynLib.open("libvulkan.so.1");
        errdefer lib.close();

        const gip = lib.lookup(VkGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return error.SymbolNotFound;
        const create_instance = try loadGlobal(VkCreateInstance, gip, "vkCreateInstance");

        return .{
            .lib = lib,
            .getInstanceProcAddr = gip,
            .createInstanceFn = create_instance,
            .instance = undefined,
        };
    }

    fn close(self: *Vulkan) void {
        self.lib.close();
    }

    fn createInstance(self: *Vulkan) !c.VkInstance {
        var app: c.VkApplicationInfo = std.mem.zeroes(c.VkApplicationInfo);
        app.sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "vk-zig-matmul";
        app.applicationVersion = c.VK_MAKE_VERSION(0, 1, 0);
        app.pEngineName = "none";
        app.engineVersion = c.VK_MAKE_VERSION(0, 1, 0);
        app.apiVersion = c.VK_API_VERSION_1_2;

        var info: c.VkInstanceCreateInfo = std.mem.zeroes(c.VkInstanceCreateInfo);
        info.sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;

        var instance: c.VkInstance = null;
        try vkCheck(self.createInstanceFn(&info, null, &instance));

        self.instance = .{
            .destroyInstance = try loadInstance(VkDestroyInstance, self.getInstanceProcAddr, instance, "vkDestroyInstance"),
            .enumeratePhysicalDevices = try loadInstance(VkEnumeratePhysicalDevices, self.getInstanceProcAddr, instance, "vkEnumeratePhysicalDevices"),
            .getPhysicalDeviceProperties = try loadInstance(VkGetPhysicalDeviceProperties, self.getInstanceProcAddr, instance, "vkGetPhysicalDeviceProperties"),
            .getPhysicalDeviceMemoryProperties = try loadInstance(VkGetPhysicalDeviceMemoryProperties, self.getInstanceProcAddr, instance, "vkGetPhysicalDeviceMemoryProperties"),
            .getPhysicalDeviceQueueFamilyProperties = try loadInstance(VkGetPhysicalDeviceQueueFamilyProperties, self.getInstanceProcAddr, instance, "vkGetPhysicalDeviceQueueFamilyProperties"),
            .enumerateDeviceExtensionProperties = try loadInstance(VkEnumerateDeviceExtensionProperties, self.getInstanceProcAddr, instance, "vkEnumerateDeviceExtensionProperties"),
            .getCooperativeMatrixPropertiesKHR = loadInstanceOptional(VkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, self.getInstanceProcAddr, instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"),
            .createDevice = try loadInstance(VkCreateDevice, self.getInstanceProcAddr, instance, "vkCreateDevice"),
            .getDeviceProcAddr = try loadInstance(VkGetDeviceProcAddr, self.getInstanceProcAddr, instance, "vkGetDeviceProcAddr"),
        };
        return instance;
    }
};

fn loadGlobal(comptime T: type, gip: VkGetInstanceProcAddr, name: [*:0]const u8) !T {
    const raw = gip(null, name) orelse return error.SymbolNotFound;
    return @ptrCast(raw);
}

fn loadInstance(comptime T: type, gip: VkGetInstanceProcAddr, instance: c.VkInstance, name: [*:0]const u8) !T {
    const raw = gip(instance, name) orelse return error.SymbolNotFound;
    return @ptrCast(raw);
}

fn loadInstanceOptional(comptime T: type, gip: VkGetInstanceProcAddr, instance: c.VkInstance, name: [*:0]const u8) ?T {
    const raw = gip(instance, name) orelse return null;
    return @ptrCast(raw);
}

fn loadDevice(comptime T: type, gdp: VkGetDeviceProcAddr, device: c.VkDevice, name: [*:0]const u8) !T {
    const raw = gdp(device, name) orelse return error.SymbolNotFound;
    return @ptrCast(raw);
}

const SelectedDevice = struct {
    physical_device: c.VkPhysicalDevice,
    queue_family: u32,
};

fn selectPhysicalDevice(vk: *Vulkan, allocator: std.mem.Allocator, instance: c.VkInstance, device_index: ?usize, device_substr: ?[]const u8, list_only: bool) !SelectedDevice {
    var count: u32 = 0;
    try vkCheck(vk.instance.enumeratePhysicalDevices(instance, &count, null));
    if (count == 0) return error.NoVulkanDevice;
    if (device_index) |index| {
        if (index >= @as(usize, count)) {
            log.err("--device={d} is out of range; use --list-devices to see available Vulkan devices", .{index});
            std.process.exit(2);
        }
    }

    const devices = try allocator.alloc(c.VkPhysicalDevice, count);
    defer allocator.free(devices);
    try vkCheck(vk.instance.enumeratePhysicalDevices(instance, &count, devices.ptr));

    var selected: ?SelectedDevice = null;
    for (devices[0..count], 0..) |physical_device, i| {
        var props: c.VkPhysicalDeviceProperties = undefined;
        vk.instance.getPhysicalDeviceProperties(physical_device, &props);
        const name = deviceName(&props);
        const queue_family = findComputeQueueFamily(vk, allocator, physical_device) catch null;

        std.debug.print("device[{d}]: {s} type={d} api={d}.{d}.{d}{s}\n", .{
            i,
            name,
            deviceType(&props),
            c.VK_VERSION_MAJOR(apiVersion(&props)),
            c.VK_VERSION_MINOR(apiVersion(&props)),
            c.VK_VERSION_PATCH(apiVersion(&props)),
            if (queue_family == null) " no-compute-queue" else "",
        });

        if (list_only or queue_family == null) continue;
        if (device_index) |index| {
            if (i != index) continue;
        }
        if (device_substr) |needle| {
            if (std.mem.indexOf(u8, name, needle) == null) continue;
        } else if (deviceType(&props) == c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
            continue;
        }
        if (selected == null) {
            selected = .{ .physical_device = physical_device, .queue_family = queue_family.? };
        }
    }

    if (list_only) return .{ .physical_device = devices[0], .queue_family = 0 };
    return selected orelse error.NoMatchingDevice;
}

fn findComputeQueueFamily(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: c.VkPhysicalDevice) !?u32 {
    var count: u32 = 0;
    vk.instance.getPhysicalDeviceQueueFamilyProperties(physical_device, &count, null);
    if (count == 0) return null;

    const families = try allocator.alloc(c.VkQueueFamilyProperties, count);
    defer allocator.free(families);
    vk.instance.getPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.ptr);

    for (families[0..count], 0..) |family, i| {
        if ((family.queueFlags & c.VK_QUEUE_COMPUTE_BIT) != 0) return @intCast(i);
    }
    return null;
}

fn validateCoopDimensions(opts: Options) !void {
    const tile_n = opts.shader.tileN();
    if (opts.m % 16 == 0 and opts.n % tile_n == 0 and opts.k % 16 == 0) return;
    log.err("--shader={s} requires m % 16 == 0, n % {d} == 0, and k % 16 == 0; use --shader=zig for arbitrary dimensions", .{ opts.shader.name(), tile_n });
    return error.InvalidCoopDimensions;
}

fn requireDeviceExtensions(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: c.VkPhysicalDevice, shader: ShaderMode) !void {
    if (!shader.isCoop()) return;

    var count: u32 = 0;
    try vkCheck(vk.instance.enumerateDeviceExtensionProperties(physical_device, null, &count, null));
    const extensions = try allocator.alloc(c.VkExtensionProperties, count);
    defer allocator.free(extensions);
    try vkCheck(vk.instance.enumerateDeviceExtensionProperties(physical_device, null, &count, extensions.ptr));

    if (!hasDeviceExtension(extensions, c.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        log.err("{s} requires {s}, but the selected device does not advertise it", .{ shader.name(), c.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME });
        return error.RequiredDeviceExtensionMissing;
    }
    if (shader == .coop_bf16 and !hasDeviceExtension(extensions, c.VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME)) {
        log.err("{s} requires {s}, but the selected device does not advertise it", .{ shader.name(), c.VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME });
        return error.RequiredDeviceExtensionMissing;
    }
}

fn hasDeviceExtension(extensions: []const c.VkExtensionProperties, needle: []const u8) bool {
    for (extensions) |extension| {
        const name = std.mem.sliceTo(&extension.extensionName, 0);
        if (std.mem.eql(u8, name, needle)) return true;
    }
    return false;
}

fn requireCoopMatrixProperty(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: c.VkPhysicalDevice, shader: ShaderMode) !void {
    const get_props = vk.instance.getCooperativeMatrixPropertiesKHR orelse {
        log.err("Vulkan loader does not expose vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR", .{});
        return error.RequiredDeviceExtensionMissing;
    };

    var count: u32 = 0;
    try vkCheck(get_props(physical_device, &count, null));
    if (count == 0) {
        log.err("{s} requires a 16x8x16 subgroup cooperative-matrix property; selected device returned none", .{shader.name()});
        return error.RequiredCoopMatrixPropertyMissing;
    }

    const props = try allocator.alloc(c.VkCooperativeMatrixPropertiesKHR, count);
    defer allocator.free(props);
    for (props) |*prop| {
        prop.* = std.mem.zeroes(c.VkCooperativeMatrixPropertiesKHR);
        prop.sType = c.VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    }
    try vkCheck(get_props(physical_device, &count, props.ptr));

    const input_type = shader.inputComponentType();
    for (props[0..count]) |prop| {
        if (prop.MSize == 16 and prop.NSize == @as(u32, @intCast(shader.tileN())) and prop.KSize == 16 and
            prop.AType == input_type and prop.BType == input_type and
            prop.CType == c.VK_COMPONENT_TYPE_FLOAT32_KHR and
            prop.ResultType == c.VK_COMPONENT_TYPE_FLOAT32_KHR and
            prop.saturatingAccumulation == c.VK_FALSE and
            prop.scope == c.VK_SCOPE_SUBGROUP_KHR)
        {
            return;
        }
    }

    log.err("selected device lacks required {s} cooperative matrix property: M=16 N={d} K=16 A/B={d} C/Result=FLOAT32 scope=SUBGROUP", .{ shader.name(), shader.tileN(), input_type });
    for (props[0..@min(count, 24)]) |prop| {
        log.err("  property: M={d} N={d} K={d} A={d} B={d} C={d} Result={d} sat={d} scope={d}", .{
            prop.MSize,
            prop.NSize,
            prop.KSize,
            prop.AType,
            prop.BType,
            prop.CType,
            prop.ResultType,
            prop.saturatingAccumulation,
            prop.scope,
        });
    }
    return error.RequiredCoopMatrixPropertyMissing;
}

const Device = struct {
    handle: c.VkDevice,
    fns: DeviceFns,
    queue: c.VkQueue,
    physical_device: c.VkPhysicalDevice,
    queue_family: u32,

    fn deinit(self: *Device) void {
        _ = self.fns.deviceWaitIdle(self.handle);
        self.fns.destroyDevice(self.handle, null);
    }
};

fn createDevice(vk: *Vulkan, allocator: std.mem.Allocator, physical_device: c.VkPhysicalDevice, queue_family: u32, shader: ShaderMode) !Device {
    try requireDeviceExtensions(vk, allocator, physical_device, shader);

    var priorities = [_]f32{1.0};
    var queue_info: c.VkDeviceQueueCreateInfo = std.mem.zeroes(c.VkDeviceQueueCreateInfo);
    queue_info.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = priorities[0..].ptr;

    var device_info: c.VkDeviceCreateInfo = std.mem.zeroes(c.VkDeviceCreateInfo);
    device_info.sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    var queue_infos = [_]c.VkDeviceQueueCreateInfo{queue_info};
    device_info.pQueueCreateInfos = queue_infos[0..].ptr;

    var extensions = [_][*:0]const u8{
        c.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
        c.VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME,
    };
    if (shader.isCoop()) {
        device_info.enabledExtensionCount = if (shader == .coop_bf16) 2 else 1;
        device_info.ppEnabledExtensionNames = extensions[0..device_info.enabledExtensionCount].ptr;
    }

    var coop_features: c.VkPhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(c.VkPhysicalDeviceCooperativeMatrixFeaturesKHR);
    var storage16_features: c.VkPhysicalDevice16BitStorageFeatures = std.mem.zeroes(c.VkPhysicalDevice16BitStorageFeatures);
    var f16_features: c.VkPhysicalDeviceShaderFloat16Int8Features = std.mem.zeroes(c.VkPhysicalDeviceShaderFloat16Int8Features);
    var bf16_features: c.VkPhysicalDeviceShaderBfloat16FeaturesKHR = std.mem.zeroes(c.VkPhysicalDeviceShaderBfloat16FeaturesKHR);

    if (shader.isCoop()) {
        coop_features.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coop_features.cooperativeMatrix = c.VK_TRUE;

        storage16_features.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        storage16_features.storageBuffer16BitAccess = c.VK_TRUE;
        storage16_features.pNext = &coop_features;

        if (shader == .coop_f16) {
            f16_features.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            f16_features.shaderFloat16 = c.VK_TRUE;
            f16_features.pNext = &storage16_features;
            device_info.pNext = &f16_features;
        } else {
            bf16_features.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
            bf16_features.shaderBFloat16Type = c.VK_TRUE;
            bf16_features.shaderBFloat16CooperativeMatrix = c.VK_TRUE;
            bf16_features.pNext = &storage16_features;
            device_info.pNext = &bf16_features;
        }
    }

    var handle: c.VkDevice = null;
    const create_result = vk.instance.createDevice(physical_device, &device_info, null, &handle);
    if (create_result == c.VK_ERROR_EXTENSION_NOT_PRESENT) return error.RequiredDeviceExtensionMissing;
    if (create_result == c.VK_ERROR_FEATURE_NOT_PRESENT) return error.RequiredDeviceFeatureMissing;
    try vkCheck(create_result);

    const fns: DeviceFns = .{
        .destroyDevice = try loadDevice(VkDestroyDevice, vk.instance.getDeviceProcAddr, handle, "vkDestroyDevice"),
        .getDeviceQueue = try loadDevice(VkGetDeviceQueue, vk.instance.getDeviceProcAddr, handle, "vkGetDeviceQueue"),
        .createBuffer = try loadDevice(VkCreateBuffer, vk.instance.getDeviceProcAddr, handle, "vkCreateBuffer"),
        .destroyBuffer = try loadDevice(VkDestroyBuffer, vk.instance.getDeviceProcAddr, handle, "vkDestroyBuffer"),
        .getBufferMemoryRequirements = try loadDevice(VkGetBufferMemoryRequirements, vk.instance.getDeviceProcAddr, handle, "vkGetBufferMemoryRequirements"),
        .allocateMemory = try loadDevice(VkAllocateMemory, vk.instance.getDeviceProcAddr, handle, "vkAllocateMemory"),
        .freeMemory = try loadDevice(VkFreeMemory, vk.instance.getDeviceProcAddr, handle, "vkFreeMemory"),
        .bindBufferMemory = try loadDevice(VkBindBufferMemory, vk.instance.getDeviceProcAddr, handle, "vkBindBufferMemory"),
        .mapMemory = try loadDevice(VkMapMemory, vk.instance.getDeviceProcAddr, handle, "vkMapMemory"),
        .unmapMemory = try loadDevice(VkUnmapMemory, vk.instance.getDeviceProcAddr, handle, "vkUnmapMemory"),
        .createShaderModule = try loadDevice(VkCreateShaderModule, vk.instance.getDeviceProcAddr, handle, "vkCreateShaderModule"),
        .destroyShaderModule = try loadDevice(VkDestroyShaderModule, vk.instance.getDeviceProcAddr, handle, "vkDestroyShaderModule"),
        .createDescriptorSetLayout = try loadDevice(VkCreateDescriptorSetLayout, vk.instance.getDeviceProcAddr, handle, "vkCreateDescriptorSetLayout"),
        .destroyDescriptorSetLayout = try loadDevice(VkDestroyDescriptorSetLayout, vk.instance.getDeviceProcAddr, handle, "vkDestroyDescriptorSetLayout"),
        .createPipelineLayout = try loadDevice(VkCreatePipelineLayout, vk.instance.getDeviceProcAddr, handle, "vkCreatePipelineLayout"),
        .destroyPipelineLayout = try loadDevice(VkDestroyPipelineLayout, vk.instance.getDeviceProcAddr, handle, "vkDestroyPipelineLayout"),
        .createComputePipelines = try loadDevice(VkCreateComputePipelines, vk.instance.getDeviceProcAddr, handle, "vkCreateComputePipelines"),
        .destroyPipeline = try loadDevice(VkDestroyPipeline, vk.instance.getDeviceProcAddr, handle, "vkDestroyPipeline"),
        .createDescriptorPool = try loadDevice(VkCreateDescriptorPool, vk.instance.getDeviceProcAddr, handle, "vkCreateDescriptorPool"),
        .destroyDescriptorPool = try loadDevice(VkDestroyDescriptorPool, vk.instance.getDeviceProcAddr, handle, "vkDestroyDescriptorPool"),
        .allocateDescriptorSets = try loadDevice(VkAllocateDescriptorSets, vk.instance.getDeviceProcAddr, handle, "vkAllocateDescriptorSets"),
        .updateDescriptorSets = try loadDevice(VkUpdateDescriptorSets, vk.instance.getDeviceProcAddr, handle, "vkUpdateDescriptorSets"),
        .createCommandPool = try loadDevice(VkCreateCommandPool, vk.instance.getDeviceProcAddr, handle, "vkCreateCommandPool"),
        .destroyCommandPool = try loadDevice(VkDestroyCommandPool, vk.instance.getDeviceProcAddr, handle, "vkDestroyCommandPool"),
        .allocateCommandBuffers = try loadDevice(VkAllocateCommandBuffers, vk.instance.getDeviceProcAddr, handle, "vkAllocateCommandBuffers"),
        .beginCommandBuffer = try loadDevice(VkBeginCommandBuffer, vk.instance.getDeviceProcAddr, handle, "vkBeginCommandBuffer"),
        .endCommandBuffer = try loadDevice(VkEndCommandBuffer, vk.instance.getDeviceProcAddr, handle, "vkEndCommandBuffer"),
        .cmdBindPipeline = try loadDevice(VkCmdBindPipeline, vk.instance.getDeviceProcAddr, handle, "vkCmdBindPipeline"),
        .cmdBindDescriptorSets = try loadDevice(VkCmdBindDescriptorSets, vk.instance.getDeviceProcAddr, handle, "vkCmdBindDescriptorSets"),
        .cmdPushConstants = try loadDevice(VkCmdPushConstants, vk.instance.getDeviceProcAddr, handle, "vkCmdPushConstants"),
        .cmdCopyBuffer = try loadDevice(VkCmdCopyBuffer, vk.instance.getDeviceProcAddr, handle, "vkCmdCopyBuffer"),
        .cmdPipelineBarrier = try loadDevice(VkCmdPipelineBarrier, vk.instance.getDeviceProcAddr, handle, "vkCmdPipelineBarrier"),
        .cmdDispatch = try loadDevice(VkCmdDispatch, vk.instance.getDeviceProcAddr, handle, "vkCmdDispatch"),
        .createFence = try loadDevice(VkCreateFence, vk.instance.getDeviceProcAddr, handle, "vkCreateFence"),
        .destroyFence = try loadDevice(VkDestroyFence, vk.instance.getDeviceProcAddr, handle, "vkDestroyFence"),
        .queueSubmit = try loadDevice(VkQueueSubmit, vk.instance.getDeviceProcAddr, handle, "vkQueueSubmit"),
        .waitForFences = try loadDevice(VkWaitForFences, vk.instance.getDeviceProcAddr, handle, "vkWaitForFences"),
        .resetFences = try loadDevice(VkResetFences, vk.instance.getDeviceProcAddr, handle, "vkResetFences"),
        .deviceWaitIdle = try loadDevice(VkDeviceWaitIdle, vk.instance.getDeviceProcAddr, handle, "vkDeviceWaitIdle"),
    };

    var queue: c.VkQueue = null;
    fns.getDeviceQueue(handle, queue_family, 0, &queue);

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

    var a = try createBuffer(vk, device, a_len * @sizeOf(f32), c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    defer destroyBuffer(device, a);
    var b = try createBuffer(vk, device, b_len * @sizeOf(f32), c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    defer destroyBuffer(device, b);
    var out = try createBuffer(vk, device, c_len * @sizeOf(f32), c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    defer destroyBuffer(device, out);

    const a_values = a.slice(f32);
    const b_values = b.slice(f32);
    const out_values = out.slice(f32);
    fillInputA(a_values, opts.m, opts.k);
    fillInputB(b_values, opts.k, opts.n);
    @memset(out_values, 0);

    const shader_module = try createShaderModule(device, opts.shader);
    defer device.fns.destroyShaderModule(device.handle, shader_module, null);

    const descriptor_layout = try createDescriptorSetLayout(device);
    defer device.fns.destroyDescriptorSetLayout(device.handle, descriptor_layout, null);

    const pipeline_layout = try createPipelineLayout(device, descriptor_layout);
    defer device.fns.destroyPipelineLayout(device.handle, pipeline_layout, null);

    const pipeline = try createPipeline(device, pipeline_layout, shader_module);
    defer device.fns.destroyPipeline(device.handle, pipeline, null);

    const descriptor_pool = try createDescriptorPool(device);
    defer device.fns.destroyDescriptorPool(device.handle, descriptor_pool, null);

    const descriptor_set = try allocateDescriptorSet(device, descriptor_pool, descriptor_layout);
    updateDescriptorSet(device, descriptor_set, a, b, out);

    const command_pool = try createCommandPool(device);
    defer device.fns.destroyCommandPool(device.handle, command_pool, null);

    const command_buffer = try allocateCommandBuffer(device, command_pool);
    try recordCommands(device, command_buffer, pipeline, pipeline_layout, descriptor_set, opts);

    const fence = try createFence(device);
    defer device.fns.destroyFence(device.handle, fence, null);

    const avg_ns = try timeRepeatedDispatch(device, command_buffer, fence, opts.warmup, opts.iters);

    try validate(out_values, a_values, b_values, opts.m, opts.n, opts.k);
    printResult(opts, avg_ns);
}

fn runCoopMatmul(vk: *Vulkan, device: *Device, opts: Options) !void {
    const a_len = opts.m * opts.k;
    const b_len = opts.k * opts.n;
    const c_len = opts.m * opts.n;

    var a_stage = try createBuffer(vk, device, a_len * @sizeOf(u16), c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    defer destroyBuffer(device, a_stage);
    var b_stage = try createBuffer(vk, device, b_len * @sizeOf(u16), c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    defer destroyBuffer(device, b_stage);
    var out_stage = try createBuffer(vk, device, c_len * @sizeOf(f32), c.VK_BUFFER_USAGE_TRANSFER_DST_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    defer destroyBuffer(device, out_stage);

    const a_dev = try createBuffer(vk, device, a_stage.bytes(), c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    defer destroyBuffer(device, a_dev);
    const b_dev = try createBuffer(vk, device, b_stage.bytes(), c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    defer destroyBuffer(device, b_dev);
    const out_dev = try createBuffer(vk, device, out_stage.bytes(), c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    defer destroyBuffer(device, out_dev);

    const a_bits = a_stage.slice(u16);
    const b_bits = b_stage.slice(u16);
    const out_values = out_stage.slice(f32);
    fillInputAEncoded(a_bits, opts.m, opts.k, opts.shader);
    fillInputBEncoded(b_bits, opts.k, opts.n, opts.shader);
    @memset(out_values, 0);

    const shader_module = try createShaderModule(device, opts.shader);
    defer device.fns.destroyShaderModule(device.handle, shader_module, null);

    const descriptor_layout = try createDescriptorSetLayout(device);
    defer device.fns.destroyDescriptorSetLayout(device.handle, descriptor_layout, null);

    const pipeline_layout = try createPipelineLayout(device, descriptor_layout);
    defer device.fns.destroyPipelineLayout(device.handle, pipeline_layout, null);

    const pipeline = try createPipeline(device, pipeline_layout, shader_module);
    defer device.fns.destroyPipeline(device.handle, pipeline, null);

    const descriptor_pool = try createDescriptorPool(device);
    defer device.fns.destroyDescriptorPool(device.handle, descriptor_pool, null);

    const descriptor_set = try allocateDescriptorSet(device, descriptor_pool, descriptor_layout);
    updateDescriptorSet(device, descriptor_set, a_dev, b_dev, out_dev);

    const command_pool = try createCommandPool(device);
    defer device.fns.destroyCommandPool(device.handle, command_pool, null);

    const upload_cmd = try allocateCommandBuffer(device, command_pool);
    const compute_cmd = try allocateCommandBuffer(device, command_pool);
    const download_cmd = try allocateCommandBuffer(device, command_pool);
    try recordUploadCommands(device, upload_cmd, a_stage, a_dev, b_stage, b_dev);
    try recordCommands(device, compute_cmd, pipeline, pipeline_layout, descriptor_set, opts);
    try recordDownloadCommands(device, download_cmd, out_dev, out_stage);

    const fence = try createFence(device);
    defer device.fns.destroyFence(device.handle, fence, null);

    try submitCommand(device, upload_cmd, fence);
    const avg_ns = try timeRepeatedDispatch(device, compute_cmd, fence, opts.warmup, opts.iters);
    try submitCommand(device, download_cmd, fence);

    try validateEncoded(out_values, opts);
    printResult(opts, avg_ns);
}

fn createBuffer(vk: *Vulkan, device: *Device, byte_len: usize, usage: c.VkBufferUsageFlags, required: c.VkMemoryPropertyFlags, map: bool) !Buffer {
    var info: c.VkBufferCreateInfo = std.mem.zeroes(c.VkBufferCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = byte_len;
    info.usage = usage;
    info.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;

    var handle: c.VkBuffer = null;
    try vkCheck(device.fns.createBuffer(device.handle, &info, null, &handle));
    errdefer device.fns.destroyBuffer(device.handle, handle, null);

    var reqs: c.VkMemoryRequirements = undefined;
    device.fns.getBufferMemoryRequirements(device.handle, handle, &reqs);

    const memory_type = try findMemoryType(vk, device.physical_device, reqs.memoryTypeBits, required);

    var alloc: c.VkMemoryAllocateInfo = std.mem.zeroes(c.VkMemoryAllocateInfo);
    alloc.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = memory_type;

    var memory: c.VkDeviceMemory = null;
    try vkCheck(device.fns.allocateMemory(device.handle, &alloc, null, &memory));
    errdefer device.fns.freeMemory(device.handle, memory, null);

    try vkCheck(device.fns.bindBufferMemory(device.handle, handle, memory, 0));

    var mapped: ?[*]u8 = null;
    if (map) {
        var mapped_raw: ?*anyopaque = null;
        try vkCheck(device.fns.mapMemory(device.handle, memory, 0, alloc.allocationSize, 0, &mapped_raw));
        mapped = @ptrCast(mapped_raw.?);
    }

    return .{ .handle = handle, .memory = memory, .mapped = mapped, .byte_len = byte_len };
}

fn destroyBuffer(device: *Device, buffer: Buffer) void {
    if (buffer.mapped != null) device.fns.unmapMemory(device.handle, buffer.memory);
    device.fns.destroyBuffer(device.handle, buffer.handle, null);
    device.fns.freeMemory(device.handle, buffer.memory, null);
}

fn findMemoryType(vk: *Vulkan, physical_device: c.VkPhysicalDevice, type_bits: u32, required: c.VkMemoryPropertyFlags) !u32 {
    var props: c.VkPhysicalDeviceMemoryProperties = undefined;
    vk.instance.getPhysicalDeviceMemoryProperties(physical_device, &props);

    for (props.memoryTypes[0..props.memoryTypeCount], 0..) |mem_type, i| {
        const bit: u32 = @as(u32, 1) << @intCast(i);
        if ((type_bits & bit) != 0 and (mem_type.propertyFlags & required) == required) return @intCast(i);
    }
    return error.MemoryTypeNotFound;
}

fn createShaderModule(device: *Device, shader: ShaderMode) !c.VkShaderModule {
    const words = switch (shader) {
        .zig => matmul_zig_spv.words[0..],
        .coop_bf16 => matmul_coop_bf16_spv.words[0..],
        .coop_f16 => matmul_coop_f16_spv.words[0..],
    };

    var info: c.VkShaderModuleCreateInfo = std.mem.zeroes(c.VkShaderModuleCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = words.len * @sizeOf(u32);
    info.pCode = words.ptr;

    var module: c.VkShaderModule = null;
    try vkCheck(device.fns.createShaderModule(device.handle, &info, null, &module));
    return module;
}

fn createDescriptorSetLayout(device: *Device) !c.VkDescriptorSetLayout {
    var bindings: [3]c.VkDescriptorSetLayoutBinding = undefined;
    for (&bindings, 0..) |*binding, i| {
        binding.* = std.mem.zeroes(c.VkDescriptorSetLayoutBinding);
        binding.binding = @intCast(i);
        binding.descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT;
    }

    var info: c.VkDescriptorSetLayoutCreateInfo = std.mem.zeroes(c.VkDescriptorSetLayoutCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = bindings.len;
    info.pBindings = bindings[0..].ptr;

    var layout: c.VkDescriptorSetLayout = null;
    try vkCheck(device.fns.createDescriptorSetLayout(device.handle, &info, null, &layout));
    return layout;
}

fn createPipelineLayout(device: *Device, descriptor_layout: c.VkDescriptorSetLayout) !c.VkPipelineLayout {
    var push: c.VkPushConstantRange = std.mem.zeroes(c.VkPushConstantRange);
    push.stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = @sizeOf(PushConstants);

    var info: c.VkPipelineLayoutCreateInfo = std.mem.zeroes(c.VkPipelineLayoutCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    var layouts = [_]c.VkDescriptorSetLayout{descriptor_layout};
    var ranges = [_]c.VkPushConstantRange{push};
    info.setLayoutCount = 1;
    info.pSetLayouts = layouts[0..].ptr;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = ranges[0..].ptr;

    var layout: c.VkPipelineLayout = null;
    try vkCheck(device.fns.createPipelineLayout(device.handle, &info, null, &layout));
    return layout;
}

fn createPipeline(device: *Device, pipeline_layout: c.VkPipelineLayout, shader_module: c.VkShaderModule) !c.VkPipeline {
    var stage: c.VkPipelineShaderStageCreateInfo = std.mem.zeroes(c.VkPipelineShaderStageCreateInfo);
    stage.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = c.VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader_module;
    stage.pName = "main";

    var info: c.VkComputePipelineCreateInfo = std.mem.zeroes(c.VkComputePipelineCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage = stage;
    info.layout = pipeline_layout;

    var infos = [_]c.VkComputePipelineCreateInfo{info};
    var pipelines = [_]c.VkPipeline{null};
    try vkCheck(device.fns.createComputePipelines(device.handle, null, 1, infos[0..].ptr, null, pipelines[0..].ptr));
    return pipelines[0];
}

fn createDescriptorPool(device: *Device) !c.VkDescriptorPool {
    var size: c.VkDescriptorPoolSize = std.mem.zeroes(c.VkDescriptorPoolSize);
    size.type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 3;

    var info: c.VkDescriptorPoolCreateInfo = std.mem.zeroes(c.VkDescriptorPoolCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = 1;
    info.poolSizeCount = 1;
    var sizes = [_]c.VkDescriptorPoolSize{size};
    info.pPoolSizes = sizes[0..].ptr;

    var pool: c.VkDescriptorPool = null;
    try vkCheck(device.fns.createDescriptorPool(device.handle, &info, null, &pool));
    return pool;
}

fn allocateDescriptorSet(device: *Device, pool: c.VkDescriptorPool, layout: c.VkDescriptorSetLayout) !c.VkDescriptorSet {
    var info: c.VkDescriptorSetAllocateInfo = std.mem.zeroes(c.VkDescriptorSetAllocateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool;
    info.descriptorSetCount = 1;
    var layouts = [_]c.VkDescriptorSetLayout{layout};
    info.pSetLayouts = layouts[0..].ptr;

    var sets = [_]c.VkDescriptorSet{null};
    try vkCheck(device.fns.allocateDescriptorSets(device.handle, &info, sets[0..].ptr));
    return sets[0];
}

fn updateDescriptorSet(device: *Device, set: c.VkDescriptorSet, a: Buffer, b: Buffer, out: Buffer) void {
    var infos = [_]c.VkDescriptorBufferInfo{
        .{ .buffer = a.handle, .offset = 0, .range = a.bytes() },
        .{ .buffer = b.handle, .offset = 0, .range = b.bytes() },
        .{ .buffer = out.handle, .offset = 0, .range = out.bytes() },
    };
    var writes: [3]c.VkWriteDescriptorSet = undefined;
    for (&writes, 0..) |*write, i| {
        write.* = std.mem.zeroes(c.VkWriteDescriptorSet);
        write.sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = @intCast(i);
        write.descriptorCount = 1;
        write.descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = @ptrCast(&infos[i]);
    }
    device.fns.updateDescriptorSets(device.handle, writes.len, &writes, 0, null);
}

fn createCommandPool(device: *Device) !c.VkCommandPool {
    var info: c.VkCommandPoolCreateInfo = std.mem.zeroes(c.VkCommandPoolCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.queueFamilyIndex = device.queue_family;

    var pool: c.VkCommandPool = null;
    try vkCheck(device.fns.createCommandPool(device.handle, &info, null, &pool));
    return pool;
}

fn allocateCommandBuffer(device: *Device, pool: c.VkCommandPool) !c.VkCommandBuffer {
    var info: c.VkCommandBufferAllocateInfo = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool;
    info.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;

    var buffers = [_]c.VkCommandBuffer{null};
    try vkCheck(device.fns.allocateCommandBuffers(device.handle, &info, buffers[0..].ptr));
    return buffers[0];
}

fn recordCommands(device: *Device, command_buffer: c.VkCommandBuffer, pipeline: c.VkPipeline, pipeline_layout: c.VkPipelineLayout, descriptor_set: c.VkDescriptorSet, opts: Options) !void {
    var begin: c.VkCommandBufferBeginInfo = std.mem.zeroes(c.VkCommandBufferBeginInfo);
    begin.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    try vkCheck(device.fns.beginCommandBuffer(command_buffer, &begin));

    const pc: PushConstants = .{
        .m = @intCast(opts.m),
        .n = @intCast(opts.n),
        .k = @intCast(opts.k),
        .a_stride = @intCast(opts.k),
        .b_stride = @intCast(opts.n),
        .c_stride = @intCast(opts.n),
    };

    device.fns.cmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    var sets = [_]c.VkDescriptorSet{descriptor_set};
    device.fns.cmdBindDescriptorSets(command_buffer, c.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, sets[0..].ptr, 0, null);
    device.fns.cmdPushConstants(command_buffer, pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConstants), &pc);
    const dispatch_x: u32 = if (opts.shader.isCoop()) @intCast(opts.n / opts.shader.tileN()) else roundUpDiv(@intCast(opts.n), 16);
    const dispatch_y: u32 = if (opts.shader.isCoop()) @intCast(opts.m / 16) else roundUpDiv(@intCast(opts.m), 16);
    device.fns.cmdDispatch(command_buffer, dispatch_x, dispatch_y, 1);

    try vkCheck(device.fns.endCommandBuffer(command_buffer));
}

fn recordUploadCommands(device: *Device, command_buffer: c.VkCommandBuffer, a_stage: Buffer, a_dev: Buffer, b_stage: Buffer, b_dev: Buffer) !void {
    var begin: c.VkCommandBufferBeginInfo = std.mem.zeroes(c.VkCommandBufferBeginInfo);
    begin.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    try vkCheck(device.fns.beginCommandBuffer(command_buffer, &begin));

    copyBuffer(device, command_buffer, a_stage, a_dev);
    copyBuffer(device, command_buffer, b_stage, b_dev);

    var barriers = [_]c.VkBufferMemoryBarrier{
        bufferBarrier(a_dev, c.VK_ACCESS_TRANSFER_WRITE_BIT, c.VK_ACCESS_SHADER_READ_BIT),
        bufferBarrier(b_dev, c.VK_ACCESS_TRANSFER_WRITE_BIT, c.VK_ACCESS_SHADER_READ_BIT),
    };
    device.fns.cmdPipelineBarrier(command_buffer, c.VK_PIPELINE_STAGE_TRANSFER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, barriers.len, barriers[0..].ptr, 0, null);

    try vkCheck(device.fns.endCommandBuffer(command_buffer));
}

fn recordDownloadCommands(device: *Device, command_buffer: c.VkCommandBuffer, out_dev: Buffer, out_stage: Buffer) !void {
    var begin: c.VkCommandBufferBeginInfo = std.mem.zeroes(c.VkCommandBufferBeginInfo);
    begin.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    try vkCheck(device.fns.beginCommandBuffer(command_buffer, &begin));

    var barriers = [_]c.VkBufferMemoryBarrier{
        bufferBarrier(out_dev, c.VK_ACCESS_SHADER_WRITE_BIT, c.VK_ACCESS_TRANSFER_READ_BIT),
    };
    device.fns.cmdPipelineBarrier(command_buffer, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, null, barriers.len, barriers[0..].ptr, 0, null);
    copyBuffer(device, command_buffer, out_dev, out_stage);

    try vkCheck(device.fns.endCommandBuffer(command_buffer));
}

fn copyBuffer(device: *Device, command_buffer: c.VkCommandBuffer, src: Buffer, dst: Buffer) void {
    var regions = [_]c.VkBufferCopy{
        .{ .srcOffset = 0, .dstOffset = 0, .size = src.bytes() },
    };
    device.fns.cmdCopyBuffer(command_buffer, src.handle, dst.handle, 1, regions[0..].ptr);
}

fn bufferBarrier(buffer: Buffer, src_access: c.VkAccessFlags, dst_access: c.VkAccessFlags) c.VkBufferMemoryBarrier {
    return .{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = null,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer.handle,
        .offset = 0,
        .size = buffer.bytes(),
    };
}

fn createFence(device: *Device) !c.VkFence {
    var info: c.VkFenceCreateInfo = std.mem.zeroes(c.VkFenceCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    var fence: c.VkFence = null;
    try vkCheck(device.fns.createFence(device.handle, &info, null, &fence));
    return fence;
}

fn timeRepeatedDispatch(device: *Device, command_buffer: c.VkCommandBuffer, fence: c.VkFence, warmup: usize, iters: usize) !f64 {
    for (0..warmup) |_| try submitCommand(device, command_buffer, fence);

    const start = try nanoTimestamp();
    for (0..iters) |_| try submitCommand(device, command_buffer, fence);
    const end = try nanoTimestamp();

    const elapsed: f64 = @floatFromInt(end - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn nanoTimestamp() !i128 {
    var ts: Timespec = undefined;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return error.ClockFailure;
    return @as(i128, ts.tv_sec) * 1_000_000_000 + ts.tv_nsec;
}

fn submitCommand(device: *Device, command_buffer: c.VkCommandBuffer, fence: c.VkFence) !void {
    var submit: c.VkSubmitInfo = std.mem.zeroes(c.VkSubmitInfo);
    submit.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    var command_buffers = [_]c.VkCommandBuffer{command_buffer};
    submit.pCommandBuffers = command_buffers[0..].ptr;
    var submits = [_]c.VkSubmitInfo{submit};
    try vkCheck(device.fns.queueSubmit(device.queue, 1, submits[0..].ptr, fence));
    var fences = [_]c.VkFence{fence};
    try vkCheck(device.fns.waitForFences(device.handle, 1, fences[0..].ptr, c.VK_TRUE, std.math.maxInt(u64)));
    try vkCheck(device.fns.resetFences(device.handle, 1, fences[0..].ptr));
}

fn printResult(opts: Options, avg_ns: f64) void {
    const ops: f64 = 2.0 * @as(f64, @floatFromInt(opts.m)) * @as(f64, @floatFromInt(opts.n)) * @as(f64, @floatFromInt(opts.k));
    const tflops = ops / avg_ns / 1.0e3;
    std.debug.print("validation passed: shader={s} m={d} n={d} k={d} avg_ns={d:.2} TFLOP/s={d:.4}\n", .{
        opts.shader.name(),
        opts.m,
        opts.n,
        opts.k,
        avg_ns,
        tflops,
    });
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
    const abs_tol: f32 = if (opts.shader == .coop_f16) 1e-2 else 2e-1;
    const rel_tol: f32 = if (opts.shader == .coop_f16) 5e-3 else 2e-2;
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
        .coop_f16 => f32ToF16Bits(value),
        .coop_bf16 => f32ToBf16Bits(value),
        .zig => unreachable,
    };
}

fn decodeInput(bits: u16, shader: ShaderMode) f32 {
    return switch (shader) {
        .coop_f16 => f16BitsToF32(bits),
        .coop_bf16 => bf16BitsToF32(bits),
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

fn deviceName(props: *const c.VkPhysicalDeviceProperties) []const u8 {
    const name_ptr: [*:0]const u8 = @ptrCast(&props.bytes[20]);
    return std.mem.span(name_ptr);
}

fn apiVersion(props: *const c.VkPhysicalDeviceProperties) u32 {
    return std.mem.readInt(u32, props.bytes[0..4], .little);
}

fn deviceType(props: *const c.VkPhysicalDeviceProperties) u32 {
    return std.mem.readInt(u32, props.bytes[16..20], .little);
}

fn vkCheck(result: c.VkResult) !void {
    if (result == c.VK_SUCCESS) return;
    log.err("Vulkan error: {d}", .{result});
    return error.VulkanFailure;
}

fn tryLoadDeviceWaitIdle(gdp: VkGetDeviceProcAddr, device: c.VkDevice) !VkDeviceWaitIdle {
    return loadDevice(VkDeviceWaitIdle, gdp, device, "vkDeviceWaitIdle");
}
