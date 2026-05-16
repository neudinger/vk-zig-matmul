const std = @import("std");

const OpEntryPoint: u16 = 15;
const OpExecutionMode: u16 = 16;
const OpName: u16 = 5;
const OpDecorate: u16 = 71;
const OpTypePointer: u16 = 32;
const OpVariable: u16 = 59;

const DecorationBlock: u32 = 2;
const DecorationBinding: u32 = 33;
const DecorationDescriptorSet: u32 = 34;
const ExecutionModeLocalSize: u32 = 17;
const StorageClassStorageBuffer: u32 = 12;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);
    if (args.len != 3 and args.len != 5) return error.InvalidArguments;

    const input_path = args[1];
    const spv_output_path = args[2];
    const zig_output_path = if (args.len == 5) args[3] else null;
    const zig_import_name = if (args.len == 5) args[4] else null;

    const input_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, input_path, gpa, .limited(16 * 1024 * 1024));
    defer gpa.free(input_bytes);
    if (input_bytes.len % 4 != 0) return error.InvalidSpirvSize;

    const input_words = try gpa.alloc(u32, input_bytes.len / 4);
    defer gpa.free(input_words);
    for (input_words, 0..) |*word, i| {
        word.* = std.mem.readInt(u32, input_bytes[i * 4 ..][0..4], .little);
    }

    const output_words = try patchSpirv(gpa, input_words);
    defer gpa.free(output_words);
    const output_bytes = std.mem.sliceAsBytes(output_words);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = spv_output_path, .data = output_bytes });

    if (zig_output_path) |path| {
        const zig = try renderZigWords(gpa, zig_import_name.?, output_words);
        defer gpa.free(zig);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = zig });
    }
}

fn patchSpirv(gpa: std.mem.Allocator, input: []const u32) ![]u32 {
    if (input.len < 5 or input[0] != 0x07230203) return error.InvalidSpirv;

    const bound = input[3];
    const names = try gpa.alloc(?[]const u8, bound);
    defer gpa.free(names);
    @memset(names, null);

    const var_types = try gpa.alloc(u32, bound);
    defer gpa.free(var_types);
    @memset(var_types, 0);

    const pointer_pointees = try gpa.alloc(u32, bound);
    defer gpa.free(pointer_pointees);
    @memset(pointer_pointees, 0);

    var entry_id: u32 = 0;
    var first_type_word: usize = 0;

    var i: usize = 5;
    while (i < input.len) {
        const word = input[i];
        const word_count: usize = @intCast(word >> 16);
        const opcode: u16 = @intCast(word & 0xffff);
        if (word_count == 0 or i + word_count > input.len) return error.InvalidSpirv;

        switch (opcode) {
            OpEntryPoint => entry_id = input[i + 2],
            OpName => {
                const target = input[i + 1];
                if (target < bound) names[target] = decodeString(input[i + 2 .. i + word_count]);
            },
            OpTypePointer => {
                const result_id = input[i + 1];
                const storage_class = input[i + 2];
                if (result_id < bound and storage_class == StorageClassStorageBuffer) {
                    pointer_pointees[result_id] = input[i + 3];
                }
            },
            OpVariable => {
                const result_type = input[i + 1];
                const result_id = input[i + 2];
                if (result_id < bound) var_types[result_id] = result_type;
            },
            else => {},
        }

        if (first_type_word == 0 and opcode >= 19 and opcode <= 39) first_type_word = i;
        i += word_count;
    }

    if (entry_id == 0 or first_type_word == 0) return error.InvalidSpirv;

    const a_id = try findNamedId(names, "a_buffer");
    const b_id = try findNamedId(names, "b_buffer");
    const c_id = try findNamedId(names, "c_buffer");
    const buffer_struct_id = pointer_pointees[var_types[a_id]];
    if (buffer_struct_id == 0) return error.InvalidSpirv;

    const extra_words = 6 + 3 + 6 * 4;
    const output = try gpa.alloc(u32, input.len + extra_words);
    var out_i: usize = 0;

    @memcpy(output[out_i..][0..5], input[0..5]);
    out_i += 5;

    i = 5;
    while (i < input.len) {
        const word = input[i];
        const word_count: usize = @intCast(word >> 16);
        const opcode: u16 = @intCast(word & 0xffff);

        if (i == first_type_word) {
            out_i = emitDecorate(output, out_i, buffer_struct_id, DecorationBlock, null);
            out_i = emitDecorate(output, out_i, a_id, DecorationDescriptorSet, 0);
            out_i = emitDecorate(output, out_i, a_id, DecorationBinding, 0);
            out_i = emitDecorate(output, out_i, b_id, DecorationDescriptorSet, 0);
            out_i = emitDecorate(output, out_i, b_id, DecorationBinding, 1);
            out_i = emitDecorate(output, out_i, c_id, DecorationDescriptorSet, 0);
            out_i = emitDecorate(output, out_i, c_id, DecorationBinding, 2);
        }

        @memcpy(output[out_i..][0..word_count], input[i..][0..word_count]);
        out_i += word_count;

        if (opcode == OpEntryPoint) {
            output[out_i + 0] = (@as(u32, 6) << 16) | OpExecutionMode;
            output[out_i + 1] = entry_id;
            output[out_i + 2] = ExecutionModeLocalSize;
            output[out_i + 3] = 16;
            output[out_i + 4] = 16;
            output[out_i + 5] = 1;
            out_i += 6;
        }

        i += word_count;
    }

    return try gpa.realloc(output, out_i);
}

fn emitDecorate(output: []u32, index: usize, target: u32, decoration: u32, value: ?u32) usize {
    if (value) |literal| {
        output[index + 0] = (@as(u32, 4) << 16) | OpDecorate;
        output[index + 1] = target;
        output[index + 2] = decoration;
        output[index + 3] = literal;
        return index + 4;
    }
    output[index + 0] = (@as(u32, 3) << 16) | OpDecorate;
    output[index + 1] = target;
    output[index + 2] = decoration;
    return index + 3;
}

fn findNamedId(names: []const ?[]const u8, needle: []const u8) !u32 {
    for (names, 0..) |name, i| {
        if (name) |actual| {
            if (std.mem.eql(u8, actual, needle)) return @intCast(i);
        }
    }
    return error.NameNotFound;
}

fn decodeString(words: []const u32) []const u8 {
    const bytes = std.mem.sliceAsBytes(words);
    return bytes[0..std.mem.indexOfScalar(u8, bytes, 0).?];
}

fn renderZigWords(gpa: std.mem.Allocator, import_name: []const u8, words: []const u32) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    const writer = &out.writer;
    defer out.deinit();

    try writer.print(
        \\//! Generated from `matmul_kernel.zig` by `//shaders:matmul_zig_spv`.
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
