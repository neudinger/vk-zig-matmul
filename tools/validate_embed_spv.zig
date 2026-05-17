const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);

    if (args.len != 8) return error.InvalidArguments;
    if (!std.mem.eql(u8, args[1], "--spirv-val")) return error.InvalidArguments;
    if (!std.mem.eql(u8, args[3], "--target-env")) return error.InvalidArguments;

    const spirv_val = args[2];
    const target_env = args[4];
    const input_path = args[5];
    const output_path = args[6];
    const import_name = args[7];

    const result = try std.process.run(gpa, init.io, .{
        .argv = &.{ spirv_val, "--target-env", target_env, input_path },
        .stderr_limit = .limited(1024 * 1024),
        .stdout_limit = .limited(1024 * 1024),
    });
    defer gpa.free(result.stdout);
    defer gpa.free(result.stderr);

    switch (result.term) {
        .exited => |code| {
            if (code != 0) {
                if (result.stdout.len != 0) std.debug.print("{s}", .{result.stdout});
                if (result.stderr.len != 0) std.debug.print("{s}", .{result.stderr});
                return error.SpirvValidationFailed;
            }
        },
        else => return error.SpirvValidationFailed,
    }

    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, input_path, gpa, .limited(16 * 1024 * 1024));
    defer gpa.free(bytes);
    if (bytes.len % 4 != 0) return error.InvalidSpirvSize;

    const words = try gpa.alloc(u32, bytes.len / 4);
    defer gpa.free(words);
    for (words, 0..) |*word, i| {
        word.* = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
    }

    const zig = try renderZigWords(gpa, import_name, words);
    defer gpa.free(zig);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = output_path, .data = zig });
}

fn renderZigWords(gpa: std.mem.Allocator, import_name: []const u8, words: []const u32) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    const writer = &out.writer;
    defer out.deinit();

    try writer.print(
        \\//! Generated from validated SPIR-V.
        \\//! Do not edit by hand.
        \\
        \\pub const import_name = "{s}";
        \\pub const words = [_]u32{{
        \\
    , .{import_name});

    var i: usize = 0;
    while (i < words.len) : (i += 8) {
        try writer.writeAll("    ");
        const end = @min(i + 8, words.len);
        for (words[i..end]) |word| try writer.print("0x{x:0>8}, ", .{word});
        try writer.writeByte('\n');
    }
    try writer.writeAll("};\n");

    return out.toOwnedSlice();
}
