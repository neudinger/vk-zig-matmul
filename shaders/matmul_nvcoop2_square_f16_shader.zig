const nv = @import("nvcoop2_shader.zig");

comptime {
    nv.shader(.{
        .stage = .compute,
        .env = .vulkan_1_4,
        .local_size = .{ 256, 1, 1 },
        .mode = .f16_square,
    });
}

const PushConstants = extern struct {
    m: u32,
    n: u32,
    k: u32,
    a_stride: u32,
    b_stride: u32,
    c_stride: u32,
};

extern var a_buffer: nv.StorageBuffer(f16, .{
    .set = 0,
    .binding = 0,
    .readonly = true,
});

extern var b_buffer: nv.StorageBuffer(f16, .{
    .set = 0,
    .binding = 1,
    .readonly = true,
});

extern var c_buffer: nv.StorageBuffer(f32, .{
    .set = 0,
    .binding = 2,
    .writeonly = true,
});

extern var push_constants: PushConstants addrspace(.push_constant);
extern const workgroup_id: @Vector(3, u32) addrspace(.input);

export fn main() callconv(.spirv_kernel) void {
    const tile_row = workgroup_id[1] * 128;
    const tile_col = workgroup_id[0] * 128;

    var result = nv.coop.zero(.accumulator_f32_128x128);

    const layout_a = nv.tensorLayout(.a, .{
        .m = push_constants.m,
        .n = push_constants.k,
    });

    const layout_b = nv.tensorLayout(.b, .{
        .m = push_constants.k,
        .n = push_constants.n,
    });

    const layout_c = nv.tensorLayout(.c, .{
        .m = push_constants.m,
        .n = push_constants.n,
    });

    var kk: u32 = 0;
    while (kk < push_constants.k) : (kk += 16) {
        const a = nv.coop.loadTensor(.a, layout_a.slice(
            tile_row, 128,
            kk, 16,
        ));

        const b = nv.coop.loadTensor(.b, layout_b.slice(
            kk, 16,
            tile_col, 128,
        ));

        result = nv.coop.mulAdd(a, b, result);
    }

    nv.coop.storeTensor(.c, result, layout_c.slice(
        tile_row, 128,
        tile_col, 128,
    ));
}
