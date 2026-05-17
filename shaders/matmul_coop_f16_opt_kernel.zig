const std = @import("std");
const coop = @import("coop_dsl.zig");

const shader = coop.CoopMatmulShader{
    .mode = .f16_opt,
    .import_name = "matmul_coop_f16_opt_dsl_spv",
    .body = matmulKernel,
};

pub fn main(init: std.process.Init) !void {
    try coop.emitShader(init, shader);
}

fn matmulKernel(kernel: *coop.Kernel) !void {
    try kernel.begin();
    const origin = try kernel.outputTileOrigin();

    try kernel.zeroAccumulators();
    const k_loop = try kernel.beginKLoop();
    const a_tile = try kernel.loadTileA(origin.row, k_loop.kk, k_loop.a_stride);

    var tile_i: usize = 0;
    while (tile_i < kernel.nTiles()) : (tile_i += 1) {
        const col = try kernel.tileColumn(origin.col, tile_i);
        const b_tile = try kernel.loadTileB(k_loop.kk, col, k_loop.b_stride);
        try kernel.mulAdd(tile_i, a_tile, b_tile);
    }

    try kernel.endKLoop(k_loop);
    try kernel.storeAccumulators(origin);
    try kernel.end();
}
