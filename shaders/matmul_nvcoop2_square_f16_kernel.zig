const std = @import("std");
const nv = @import("nvcoop2_dsl.zig");

const shader = nv.NvCoop2Shader{
    .mode = .f16_square,
    .import_name = "matmul_nvcoop2_square_f16_spv",
    .body = matmulKernel,
};

pub fn main(init: std.process.Init) !void {
    try nv.emitShader(init, shader);
}

fn matmulKernel(k: *nv.Kernel) !void {
    try k.begin();
    const tile = try k.workgroupTile();

    var result = try k.coop.zero(.accumulator);

    const layout_a = try k.tensorLayout(.a, .{ .m = k.push.m, .n = k.push.k });
    const layout_b = try k.tensorLayout(.b, .{ .m = k.push.k, .n = k.push.n });
    const layout_c = try k.tensorLayout(.c, .{ .m = k.push.m, .n = k.push.n });

    var loop = try k.kLoop(.{ .step = 16 });
    while (try loop.next()) |chunk_k| {
        const a = try k.coop.loadTensor(.a, layout_a.slice(tile.row, 128, chunk_k, 16));
        const b = try k.coop.loadTensor(.b, layout_b.slice(chunk_k, 16, tile.col, 128));
        result = try k.coop.mulAdd(a, b, result);
    }

    try k.coop.storeTensor(.c, result, layout_c.slice(tile.row, 128, tile.col, 128));
    try k.end();
}
