const std = @import("std");
const matmul_zig_spv = @import("matmul_zig_spv");
const c = @import("vulkan_loader.zig").c;

const log = std.log.scoped(.vk_matmul);

const VkGetInstanceProcAddr = *const fn (c.VkInstance, [*:0]const u8) callconv(.c) c.PFN_vkVoidFunction;
const VkGetDeviceProcAddr = *const fn (c.VkDevice, [*:0]const u8) callconv(.c) c.PFN_vkVoidFunction;

const VkCreateInstance = *const fn (*const c.VkInstanceCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkInstance) callconv(.c) c.VkResult;
const VkDestroyInstance = *const fn (c.VkInstance, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkEnumeratePhysicalDevices = *const fn (c.VkInstance, *u32, ?[*]c.VkPhysicalDevice) callconv(.c) c.VkResult;
const VkGetPhysicalDeviceProperties = *const fn (c.VkPhysicalDevice, *c.VkPhysicalDeviceProperties) callconv(.c) void;
const VkGetPhysicalDeviceMemoryProperties = *const fn (c.VkPhysicalDevice, *c.VkPhysicalDeviceMemoryProperties) callconv(.c) void;
const VkGetPhysicalDeviceQueueFamilyProperties = *const fn (c.VkPhysicalDevice, *u32, ?[*]c.VkQueueFamilyProperties) callconv(.c) void;
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
const VkCmdDispatch = *const fn (c.VkCommandBuffer, u32, u32, u32) callconv(.c) void;
const VkCreateFence = *const fn (c.VkDevice, *const c.VkFenceCreateInfo, ?*const c.VkAllocationCallbacks, *c.VkFence) callconv(.c) c.VkResult;
const VkDestroyFence = *const fn (c.VkDevice, c.VkFence, ?*const c.VkAllocationCallbacks) callconv(.c) void;
const VkQueueSubmit = *const fn (c.VkQueue, u32, [*]const c.VkSubmitInfo, c.VkFence) callconv(.c) c.VkResult;
const VkWaitForFences = *const fn (c.VkDevice, u32, [*]const c.VkFence, c.VkBool32, u64) callconv(.c) c.VkResult;
const VkDeviceWaitIdle = *const fn (c.VkDevice) callconv(.c) c.VkResult;

const InstanceFns = struct {
    destroyInstance: VkDestroyInstance,
    enumeratePhysicalDevices: VkEnumeratePhysicalDevices,
    getPhysicalDeviceProperties: VkGetPhysicalDeviceProperties,
    getPhysicalDeviceMemoryProperties: VkGetPhysicalDeviceMemoryProperties,
    getPhysicalDeviceQueueFamilyProperties: VkGetPhysicalDeviceQueueFamilyProperties,
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
    cmdDispatch: VkCmdDispatch,
    createFence: VkCreateFence,
    destroyFence: VkDestroyFence,
    queueSubmit: VkQueueSubmit,
    waitForFences: VkWaitForFences,
    deviceWaitIdle: VkDeviceWaitIdle,
};

const Options = struct {
    m: usize = 64,
    n: usize = 64,
    k: usize = 64,
    list_devices: bool = false,
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
    mapped: [*]f32,
    len: usize,

    fn bytes(self: Buffer) usize {
        return self.len * @sizeOf(f32);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);

    const opts = try parseArgs(args);
    if (opts.m == 0 or opts.n == 0 or opts.k == 0) return error.InvalidDimensions;
    if (opts.m > std.math.maxInt(u32) or opts.n > std.math.maxInt(u32) or opts.k > std.math.maxInt(u32)) {
        return error.InvalidDimensions;
    }

    var vk = try Vulkan.open();
    defer vk.close();

    const instance = try vk.createInstance();
    defer vk.instance.destroyInstance(instance, null);

    const selected = try selectPhysicalDevice(&vk, allocator, instance, opts.device_substr, opts.list_devices);
    if (opts.list_devices) return;

    var device = try createDevice(&vk, selected.physical_device, selected.queue_family);
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
            if (!std.mem.eql(u8, value, "zig")) return error.InvalidShader;
        } else if (std.mem.startsWith(u8, arg, "--m=")) {
            opts.m = try std.fmt.parseInt(usize, arg["--m=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--n=")) {
            opts.n = try std.fmt.parseInt(usize, arg["--n=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--k=")) {
            opts.k = try std.fmt.parseInt(usize, arg["--k=".len..], 10);
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

fn loadDevice(comptime T: type, gdp: VkGetDeviceProcAddr, device: c.VkDevice, name: [*:0]const u8) !T {
    const raw = gdp(device, name) orelse return error.SymbolNotFound;
    return @ptrCast(raw);
}

const SelectedDevice = struct {
    physical_device: c.VkPhysicalDevice,
    queue_family: u32,
};

fn selectPhysicalDevice(vk: *Vulkan, allocator: std.mem.Allocator, instance: c.VkInstance, device_substr: ?[]const u8, list_only: bool) !SelectedDevice {
    var count: u32 = 0;
    try vkCheck(vk.instance.enumeratePhysicalDevices(instance, &count, null));
    if (count == 0) return error.NoVulkanDevice;

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

fn createDevice(vk: *Vulkan, physical_device: c.VkPhysicalDevice, queue_family: u32) !Device {
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

    var handle: c.VkDevice = null;
    try vkCheck(vk.instance.createDevice(physical_device, &device_info, null, &handle));

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
        .cmdDispatch = try loadDevice(VkCmdDispatch, vk.instance.getDeviceProcAddr, handle, "vkCmdDispatch"),
        .createFence = try loadDevice(VkCreateFence, vk.instance.getDeviceProcAddr, handle, "vkCreateFence"),
        .destroyFence = try loadDevice(VkDestroyFence, vk.instance.getDeviceProcAddr, handle, "vkDestroyFence"),
        .queueSubmit = try loadDevice(VkQueueSubmit, vk.instance.getDeviceProcAddr, handle, "vkQueueSubmit"),
        .waitForFences = try loadDevice(VkWaitForFences, vk.instance.getDeviceProcAddr, handle, "vkWaitForFences"),
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
    const a_len = opts.m * opts.k;
    const b_len = opts.k * opts.n;
    const c_len = opts.m * opts.n;

    var a = try createBuffer(vk, device, a_len);
    defer destroyBuffer(device, a);
    var b = try createBuffer(vk, device, b_len);
    defer destroyBuffer(device, b);
    var out = try createBuffer(vk, device, c_len);
    defer destroyBuffer(device, out);

    fillInputA(a.mapped[0..a.len], opts.m, opts.k);
    fillInputB(b.mapped[0..b.len], opts.k, opts.n);
    @memset(out.mapped[0..out.len], 0);

    const shader_module = try createShaderModule(device);
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

    var submit: c.VkSubmitInfo = std.mem.zeroes(c.VkSubmitInfo);
    submit.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    var command_buffers = [_]c.VkCommandBuffer{command_buffer};
    submit.pCommandBuffers = command_buffers[0..].ptr;
    var submits = [_]c.VkSubmitInfo{submit};
    try vkCheck(device.fns.queueSubmit(device.queue, 1, submits[0..].ptr, fence));
    var fences = [_]c.VkFence{fence};
    try vkCheck(device.fns.waitForFences(device.handle, 1, fences[0..].ptr, c.VK_TRUE, std.math.maxInt(u64)));

    try validate(out.mapped[0..out.len], a.mapped[0..a.len], b.mapped[0..b.len], opts.m, opts.n, opts.k);
    std.debug.print("validation passed: shader=zig m={d} n={d} k={d}\n", .{ opts.m, opts.n, opts.k });
}

fn createBuffer(vk: *Vulkan, device: *Device, len: usize) !Buffer {
    var info: c.VkBufferCreateInfo = std.mem.zeroes(c.VkBufferCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = len * @sizeOf(f32);
    info.usage = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;

    var handle: c.VkBuffer = null;
    try vkCheck(device.fns.createBuffer(device.handle, &info, null, &handle));
    errdefer device.fns.destroyBuffer(device.handle, handle, null);

    var reqs: c.VkMemoryRequirements = undefined;
    device.fns.getBufferMemoryRequirements(device.handle, handle, &reqs);

    const memory_type = try findMemoryType(vk, device.physical_device, reqs.memoryTypeBits, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    var alloc: c.VkMemoryAllocateInfo = std.mem.zeroes(c.VkMemoryAllocateInfo);
    alloc.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = memory_type;

    var memory: c.VkDeviceMemory = null;
    try vkCheck(device.fns.allocateMemory(device.handle, &alloc, null, &memory));
    errdefer device.fns.freeMemory(device.handle, memory, null);

    try vkCheck(device.fns.bindBufferMemory(device.handle, handle, memory, 0));

    var mapped_raw: ?*anyopaque = null;
    try vkCheck(device.fns.mapMemory(device.handle, memory, 0, alloc.allocationSize, 0, &mapped_raw));
    const mapped: [*]f32 = @ptrCast(@alignCast(mapped_raw.?));

    return .{ .handle = handle, .memory = memory, .mapped = mapped, .len = len };
}

fn destroyBuffer(device: *Device, buffer: Buffer) void {
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

fn createShaderModule(device: *Device) !c.VkShaderModule {
    var info: c.VkShaderModuleCreateInfo = std.mem.zeroes(c.VkShaderModuleCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = matmul_zig_spv.words.len * @sizeOf(u32);
    info.pCode = matmul_zig_spv.words[0..].ptr;

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
    device.fns.cmdDispatch(command_buffer, roundUpDiv(@intCast(opts.n), 16), roundUpDiv(@intCast(opts.m), 16), 1);

    try vkCheck(device.fns.endCommandBuffer(command_buffer));
}

fn createFence(device: *Device) !c.VkFence {
    var info: c.VkFenceCreateInfo = std.mem.zeroes(c.VkFenceCreateInfo);
    info.sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    var fence: c.VkFence = null;
    try vkCheck(device.fns.createFence(device.handle, &info, null, &fence));
    return fence;
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
