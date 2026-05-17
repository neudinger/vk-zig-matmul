const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);
    if (args.len != 4) return error.InvalidArguments;

    const input_path = args[1];
    const output_path = args[2];
    const import_name = args[3];

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
        \\//! Generated from assembled SPIR-V text.
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
