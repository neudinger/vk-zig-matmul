//! Experimental high-level cooperative-matrix shader emitter.
//!
//! This keeps the original `emit_coop_spv.zig` target as the opcode-level
//! reference, while giving us a separate place to grow a shader-shaped API.

const std = @import("std");

pub const Mode = enum {
    bf16,
    f16,
    bf16_opt,
    f16_opt,
    f16_wg,

    fn importName(self: Mode) []const u8 {
        return switch (self) {
            .bf16 => "matmul_coop_bf16_dsl_spv",
            .f16 => "matmul_coop_f16_dsl_spv",
            .bf16_opt => "matmul_coop_bf16_opt_dsl_spv",
            .f16_opt => "matmul_coop_f16_opt_dsl_spv",
            .f16_wg => "matmul_coop_shared_f16_dsl_spv",
        };
    }

    fn componentLabel(self: Mode) []const u8 {
        return switch (self) {
            .bf16, .bf16_opt => "bf16",
            .f16, .f16_opt, .f16_wg => "f16",
        };
    }

    fn tileN(self: Mode) u32 {
        return switch (self) {
            .bf16, .bf16_opt => 16,
            .f16, .f16_opt, .f16_wg => 8,
        };
    }

    fn nTiles(self: Mode) u32 {
        return switch (self) {
            .bf16, .f16 => 1,
            .bf16_opt => 2,
            .f16_opt, .f16_wg => 4,
        };
    }

    fn outputTileN(self: Mode) u32 {
        return switch (self) {
            .f16_wg => 64,
            else => self.tileN() * self.nTiles(),
        };
    }

    fn isF16(self: Mode) bool {
        return switch (self) {
            .f16, .f16_opt, .f16_wg => true,
            .bf16, .bf16_opt => false,
        };
    }

    fn hasOptimizedNTiling(self: Mode) bool {
        return self.nTiles() > 1;
    }

    fn isWorkgroupTiled(self: Mode) bool {
        return self == .f16_wg;
    }

    fn localSizeX(self: Mode) u32 {
        return if (self.isWorkgroupTiled()) 256 else 32;
    }
};

const max_n_tiles = 4;

pub const CoopMatmulShader = struct {
    pub const BodyFn = *const fn (*Kernel) anyerror!void;

    mode: Mode,
    import_name: []const u8,
    body: BodyFn = referenceMatmulKernel,

    pub fn fromMode(mode: Mode) CoopMatmulShader {
        return .{ .mode = mode, .import_name = mode.importName() };
    }

    pub fn emit(self: CoopMatmulShader, builder: *Builder) ![]const u32 {
        builder.len = 0;
        builder.bound = 1;
        const ids = builder.reserveIds();

        try emitHeader(builder);
        try emitCapabilities(builder, self.mode);
        try emitEntry(builder, ids, self.mode);
        try emitNames(builder, ids, self.mode);
        try emitDecorations(builder, ids);
        try emitTypes(builder, ids, self.mode);

        var kernel = Kernel{ .builder = builder, .ids = ids, .mode = self.mode };
        try self.body(&kernel);

        builder.words[3] = builder.bound;
        return builder.words[0..builder.len];
    }
};

pub fn emitShader(init: std.process.Init, shader: CoopMatmulShader) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);

    if (args.len != 3) return error.InvalidArguments;

    var builder = Builder{};
    const words = try shader.emit(&builder);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[1], .data = std.mem.sliceAsBytes(words) });

    const zig = try renderZigWords(gpa, shader, words);
    defer gpa.free(zig);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = zig });
}

fn modeFromArg(arg: []const u8) !Mode {
    if (std.mem.eql(u8, arg, "--mode=bf16")) return .bf16;
    if (std.mem.eql(u8, arg, "--mode=f16")) return .f16;
    if (std.mem.eql(u8, arg, "--mode=bf16-opt")) return .bf16_opt;
    if (std.mem.eql(u8, arg, "--mode=f16-opt")) return .f16_opt;
    if (std.mem.eql(u8, arg, "--mode=f16-wg")) return .f16_wg;
    return error.InvalidMode;
}

fn constForU32(ids: Ids, value: u32) u32 {
    return switch (value) {
        0 => ids.u32_0,
        1 => ids.u32_1,
        2 => ids.u32_2,
        3 => ids.u32_3,
        8 => ids.u32_8,
        16 => ids.u32_16,
        24 => ids.u32_24,
        32 => ids.u32_32,
        64 => ids.u32_64,
        else => unreachable,
    };
}

fn addConstU32(builder: *Builder, ids: Ids, base: u32, value: u32) !u32 {
    if (value == 0) return base;
    return binary(builder, OpIAdd, ids.u32_ty, base, constForU32(ids, value));
}

fn u32IdForTileN(ids: Ids, value: u32) u32 {
    return constForU32(ids, value);
}

const OpName: u16 = 5;
const OpMemberName: u16 = 6;
const OpExtension: u16 = 10;
const OpMemoryModel: u16 = 14;
const OpEntryPoint: u16 = 15;
const OpExecutionMode: u16 = 16;
const OpCapability: u16 = 17;
const OpTypeVoid: u16 = 19;
const OpTypeBool: u16 = 20;
const OpTypeInt: u16 = 21;
const OpTypeFloat: u16 = 22;
const OpTypeVector: u16 = 23;
const OpTypeRuntimeArray: u16 = 29;
const OpTypeStruct: u16 = 30;
const OpTypePointer: u16 = 32;
const OpTypeFunction: u16 = 33;
const OpConstant: u16 = 43;
const OpConstantComposite: u16 = 44;
const OpFunction: u16 = 54;
const OpFunctionEnd: u16 = 56;
const OpVariable: u16 = 59;
const OpLoad: u16 = 61;
const OpStore: u16 = 62;
const OpAccessChain: u16 = 65;
const OpDecorate: u16 = 71;
const OpMemberDecorate: u16 = 72;
const OpIAdd: u16 = 128;
const OpIMul: u16 = 132;
const OpUDiv: u16 = 134;
const OpUMod: u16 = 137;
const OpULessThan: u16 = 176;
const OpLoopMerge: u16 = 246;
const OpLabel: u16 = 248;
const OpBranch: u16 = 249;
const OpBranchConditional: u16 = 250;
const OpReturn: u16 = 253;
const OpTypeCooperativeMatrixKHR: u16 = 4456;
const OpCooperativeMatrixLoadKHR: u16 = 4457;
const OpCooperativeMatrixStoreKHR: u16 = 4458;
const OpCooperativeMatrixMulAddKHR: u16 = 4459;

const CapabilityShader: u32 = 1;
const CapabilityFloat16: u32 = 9;
const CapabilityStorageBuffer16BitAccess: u32 = 4433;
const CapabilityVulkanMemoryModel: u32 = 5345;
const CapabilityCooperativeMatrixKHR: u32 = 6022;
const CapabilityBFloat16TypeKHR: u32 = 5116;
const CapabilityBFloat16CooperativeMatrixKHR: u32 = 5118;

const AddressingModelLogical: u32 = 0;
const MemoryModelVulkan: u32 = 3;
const ExecutionModelGLCompute: u32 = 5;
const ExecutionModeLocalSize: u32 = 17;

const StorageClassInput: u32 = 1;
const StorageClassFunction: u32 = 7;
const StorageClassPushConstant: u32 = 9;
const StorageClassStorageBuffer: u32 = 12;

const DecorationBlock: u32 = 2;
const DecorationArrayStride: u32 = 6;
const DecorationBuiltIn: u32 = 11;
const DecorationNonWritable: u32 = 24;
const DecorationNonReadable: u32 = 25;
const DecorationBinding: u32 = 33;
const DecorationDescriptorSet: u32 = 34;
const DecorationOffset: u32 = 35;
const BuiltInWorkgroupId: u32 = 26;

const MatrixUseA: u32 = 0;
const MatrixUseB: u32 = 1;
const MatrixUseAccumulator: u32 = 2;
const ScopeSubgroup: u32 = 3;
const CooperativeMatrixLayoutRowMajor: u32 = 0;
const MemoryAccessNone: u32 = 0;
const FunctionControlNone: u32 = 0;
const LoopControlNone: u32 = 0;
const BFloat16EncodingKHR: u32 = 0;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);

    if (args.len != 4) return error.InvalidArguments;
    const mode = try modeFromArg(args[1]);

    var builder = Builder{};
    const shader = CoopMatmulShader.fromMode(mode);
    const words = try shader.emit(&builder);
    const bytes = std.mem.sliceAsBytes(words);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = bytes });

    const zig = try renderZigWords(gpa, shader, words);
    defer gpa.free(zig);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = zig });
}

const Ids = struct {
    main: u32,
    wg: u32,
    lid: u32,
    a_var: u32,
    b_var: u32,
    c_var: u32,
    pc_var: u32,

    void_ty: u32,
    bool_ty: u32,
    u32_ty: u32,
    i32_ty: u32,
    f32_ty: u32,
    in_ty: u32,
    vec3_u32_ty: u32,
    fn_void_ty: u32,

    u32_0: u32,
    u32_1: u32,
    u32_2: u32,
    u32_3: u32,
    u32_8: u32,
    u32_16: u32,
    u32_24: u32,
    u32_32: u32,
    u32_64: u32,
    i32_0: u32,
    i32_2: u32,
    i32_3: u32,
    i32_4: u32,
    i32_5: u32,
    f32_0: u32,

    acc_ty: u32,
    a_ty: u32,
    b_ty: u32,
    acc_zero: u32,

    ptr_input_vec3_u32: u32,
    ptr_input_u32: u32,
    ptr_fn_u32: u32,
    ptr_fn_acc: u32,
    ptr_pc_struct: u32,
    ptr_pc_u32: u32,
    ptr_sb_f32: u32,
    ptr_sb_in: u32,

    array_a: u32,
    array_b: u32,
    array_c: u32,
    struct_a: u32,
    struct_b: u32,
    struct_c: u32,
    ptr_sb_struct_a: u32,
    ptr_sb_struct_b: u32,
    ptr_sb_struct_c: u32,
    pc_struct: u32,

    acc_vars: [max_n_tiles]u32,
    kk_var: u32,
};

pub const TileOrigin = struct {
    row: u32,
    col: u32,
};

pub const KLoop = struct {
    header: u32,
    merge: u32,
    continue_label: u32,
    kk: u32,
    a_stride: u32,
    b_stride: u32,
};

pub const Kernel = struct {
    builder: *Builder,
    ids: Ids,
    mode: Mode,

    pub fn nTiles(self: Kernel) usize {
        return @intCast(self.mode.nTiles());
    }

    pub fn tileN(self: Kernel) u32 {
        return self.mode.tileN();
    }

    pub fn begin(self: *Kernel) !void {
        try self.builder.emit(OpFunction, &.{ self.ids.void_ty, self.ids.main, FunctionControlNone, self.ids.fn_void_ty });
        const entry_label = self.builder.nextId();
        try self.builder.emit(OpLabel, &.{entry_label});

        var tile_i: usize = 0;
        while (tile_i < self.nTiles()) : (tile_i += 1) {
            try self.builder.emit(OpVariable, &.{ self.ids.ptr_fn_acc, self.ids.acc_vars[tile_i], StorageClassFunction });
        }
        try self.builder.emit(OpVariable, &.{ self.ids.ptr_fn_u32, self.ids.kk_var, StorageClassFunction });
    }

    pub fn outputTileOrigin(self: *Kernel) !TileOrigin {
        const output_tile_n_id = u32IdForTileN(self.ids, self.mode.outputTileN());
        const wg_x_ptr = try accessChain(self.builder, self.ids.ptr_input_u32, self.ids.wg, &.{self.ids.u32_0});
        const tile_col = try load(self.builder, self.ids.u32_ty, wg_x_ptr);
        const wg_y_ptr = try accessChain(self.builder, self.ids.ptr_input_u32, self.ids.wg, &.{self.ids.u32_1});
        const tile_row = try load(self.builder, self.ids.u32_ty, wg_y_ptr);

        if (self.mode.isWorkgroupTiled()) {
            const lid_x_ptr = try accessChain(self.builder, self.ids.ptr_input_u32, self.ids.lid, &.{self.ids.u32_0});
            const lid_x = try load(self.builder, self.ids.u32_ty, lid_x_ptr);
            const subgroup_id = try binary(self.builder, OpUDiv, self.ids.u32_ty, lid_x, self.ids.u32_32);
            const row_group = try binary(self.builder, OpUDiv, self.ids.u32_ty, subgroup_id, self.ids.u32_2);
            const col_group = try binary(self.builder, OpUMod, self.ids.u32_ty, subgroup_id, self.ids.u32_2);
            const row_base = try binary(self.builder, OpIMul, self.ids.u32_ty, tile_row, self.ids.u32_64);
            const col_base = try binary(self.builder, OpIMul, self.ids.u32_ty, tile_col, output_tile_n_id);
            const row_delta = try binary(self.builder, OpIMul, self.ids.u32_ty, row_group, self.ids.u32_16);
            const col_delta = try binary(self.builder, OpIMul, self.ids.u32_ty, col_group, self.ids.u32_32);
            return .{
                .row = try binary(self.builder, OpIAdd, self.ids.u32_ty, row_base, row_delta),
                .col = try binary(self.builder, OpIAdd, self.ids.u32_ty, col_base, col_delta),
            };
        }

        return .{
            .row = try binary(self.builder, OpIMul, self.ids.u32_ty, tile_row, self.ids.u32_16),
            .col = try binary(self.builder, OpIMul, self.ids.u32_ty, tile_col, output_tile_n_id),
        };
    }

    pub fn zeroAccumulators(self: *Kernel) !void {
        var tile_i: usize = 0;
        while (tile_i < self.nTiles()) : (tile_i += 1) {
            try self.builder.emit(OpStore, &.{ self.ids.acc_vars[tile_i], self.ids.acc_zero });
        }
    }

    pub fn beginKLoop(self: *Kernel) !KLoop {
        try self.builder.emit(OpStore, &.{ self.ids.kk_var, self.ids.u32_0 });

        const loop_header = self.builder.nextId();
        const loop_cond = self.builder.nextId();
        const loop_body = self.builder.nextId();
        const loop_merge = self.builder.nextId();
        const loop_continue = self.builder.nextId();
        try self.builder.emit(OpBranch, &.{loop_header});

        try self.builder.emit(OpLabel, &.{loop_header});
        try self.builder.emit(OpLoopMerge, &.{ loop_merge, loop_continue, LoopControlNone });
        try self.builder.emit(OpBranch, &.{loop_cond});

        try self.builder.emit(OpLabel, &.{loop_cond});
        const kk_for_cond = try load(self.builder, self.ids.u32_ty, self.ids.kk_var);
        const k_value = try self.loadPush(.k);
        const cond = try binary(self.builder, OpULessThan, self.ids.bool_ty, kk_for_cond, k_value);
        try self.builder.emit(OpBranchConditional, &.{ cond, loop_body, loop_merge });

        try self.builder.emit(OpLabel, &.{loop_body});
        return .{
            .header = loop_header,
            .merge = loop_merge,
            .continue_label = loop_continue,
            .kk = try load(self.builder, self.ids.u32_ty, self.ids.kk_var),
            .a_stride = try self.loadPush(.a_stride),
            .b_stride = try self.loadPush(.b_stride),
        };
    }

    pub fn loadTileA(self: *Kernel, row: u32, kk: u32, a_stride: u32) !u32 {
        const a_row_base = try binary(self.builder, OpIMul, self.ids.u32_ty, row, a_stride);
        const a_offset = try binary(self.builder, OpIAdd, self.ids.u32_ty, a_row_base, kk);
        const a_ptr = try accessChain(self.builder, self.ids.ptr_sb_in, self.ids.a_var, &.{ self.ids.i32_0, a_offset });
        return coopLoad(self.builder, self.ids.a_ty, a_ptr, a_stride, self.ids.i32_0);
    }

    pub fn tileColumn(self: *Kernel, origin_col: u32, tile_i: usize) !u32 {
        const n_offset = self.mode.tileN() * @as(u32, @intCast(tile_i));
        return addConstU32(self.builder, self.ids, origin_col, n_offset);
    }

    pub fn loadTileB(self: *Kernel, kk: u32, col: u32, b_stride: u32) !u32 {
        const b_row_base = try binary(self.builder, OpIMul, self.ids.u32_ty, kk, b_stride);
        const b_offset = try binary(self.builder, OpIAdd, self.ids.u32_ty, b_row_base, col);
        const b_ptr = try accessChain(self.builder, self.ids.ptr_sb_in, self.ids.b_var, &.{ self.ids.i32_0, b_offset });
        return coopLoad(self.builder, self.ids.b_ty, b_ptr, b_stride, self.ids.i32_0);
    }

    pub fn mulAdd(self: *Kernel, tile_i: usize, a_mat: u32, b_mat: u32) !void {
        const acc_in = try load(self.builder, self.ids.acc_ty, self.ids.acc_vars[tile_i]);
        const acc_out = self.builder.nextId();
        try self.builder.emit(OpCooperativeMatrixMulAddKHR, &.{ self.ids.acc_ty, acc_out, a_mat, b_mat, acc_in });
        try self.builder.emit(OpStore, &.{ self.ids.acc_vars[tile_i], acc_out });
    }

    pub fn endKLoop(self: *Kernel, loop: KLoop) !void {
        try self.builder.emit(OpBranch, &.{loop.continue_label});

        try self.builder.emit(OpLabel, &.{loop.continue_label});
        const kk_old = try load(self.builder, self.ids.u32_ty, self.ids.kk_var);
        const kk_next = try binary(self.builder, OpIAdd, self.ids.u32_ty, kk_old, self.ids.u32_16);
        try self.builder.emit(OpStore, &.{ self.ids.kk_var, kk_next });
        try self.builder.emit(OpBranch, &.{loop.header});

        try self.builder.emit(OpLabel, &.{loop.merge});
    }

    pub fn storeAccumulators(self: *Kernel, origin: TileOrigin) !void {
        const c_stride = try self.loadPush(.c_stride);
        const c_row_base = try binary(self.builder, OpIMul, self.ids.u32_ty, origin.row, c_stride);

        var tile_i: usize = 0;
        while (tile_i < self.nTiles()) : (tile_i += 1) {
            const tile_col = try self.tileColumn(origin.col, tile_i);
            const acc_final = try load(self.builder, self.ids.acc_ty, self.ids.acc_vars[tile_i]);
            const c_offset = try binary(self.builder, OpIAdd, self.ids.u32_ty, c_row_base, tile_col);
            const c_ptr = try accessChain(self.builder, self.ids.ptr_sb_f32, self.ids.c_var, &.{ self.ids.i32_0, c_offset });
            try self.builder.emit(OpCooperativeMatrixStoreKHR, &.{ c_ptr, acc_final, self.ids.i32_0, c_stride, MemoryAccessNone });
        }
    }

    pub fn end(self: *Kernel) !void {
        try self.builder.emit(OpReturn, &.{});
        try self.builder.emit(OpFunctionEnd, &.{});
    }

    fn loadPush(self: *Kernel, field: PushField) !u32 {
        const member = switch (field) {
            .k => self.ids.i32_2,
            .a_stride => self.ids.i32_3,
            .b_stride => self.ids.i32_4,
            .c_stride => self.ids.i32_5,
        };
        const ptr = try accessChain(self.builder, self.ids.ptr_pc_u32, self.ids.pc_var, &.{member});
        return load(self.builder, self.ids.u32_ty, ptr);
    }
};

const PushField = enum {
    k,
    a_stride,
    b_stride,
    c_stride,
};

fn referenceMatmulKernel(kernel: *Kernel) !void {
    try kernel.begin();
    const origin = try kernel.outputTileOrigin();

    try kernel.zeroAccumulators();
    const loop = try kernel.beginKLoop();
    const a_mat = try kernel.loadTileA(origin.row, loop.kk, loop.a_stride);

    var tile_i: usize = 0;
    while (tile_i < kernel.nTiles()) : (tile_i += 1) {
        const col = try kernel.tileColumn(origin.col, tile_i);
        const b_mat = try kernel.loadTileB(loop.kk, col, loop.b_stride);
        try kernel.mulAdd(tile_i, a_mat, b_mat);
    }

    try kernel.endKLoop(loop);
    try kernel.storeAccumulators(origin);
    try kernel.end();
}

const Builder = struct {
    words: [4096]u32 = undefined,
    len: usize = 0,
    bound: u32 = 1,

    fn nextId(self: *Builder) u32 {
        const id = self.bound;
        self.bound += 1;
        return id;
    }

    fn emit(self: *Builder, opcode: u16, operands: []const u32) !void {
        const wc = operands.len + 1;
        if (self.len + wc > self.words.len) return error.ModuleTooLarge;
        self.words[self.len] = (@as(u32, @intCast(wc)) << 16) | opcode;
        self.len += 1;
        @memcpy(self.words[self.len..][0..operands.len], operands);
        self.len += operands.len;
    }

    fn emitString(self: *Builder, opcode: u16, operands: []const u32, text: []const u8, trailing: []const u32) !void {
        const string_words = (text.len + 1 + 3) / 4;
        const wc = 1 + operands.len + string_words + trailing.len;
        if (self.len + wc > self.words.len) return error.ModuleTooLarge;
        self.words[self.len] = (@as(u32, @intCast(wc)) << 16) | opcode;
        self.len += 1;
        @memcpy(self.words[self.len..][0..operands.len], operands);
        self.len += operands.len;

        var i: usize = 0;
        while (i < string_words) : (i += 1) {
            var word: u32 = 0;
            for (0..4) |byte_i| {
                const text_i = i * 4 + byte_i;
                const byte = if (text_i < text.len) text[text_i] else 0;
                word |= @as(u32, byte) << @intCast(byte_i * 8);
            }
            self.words[self.len] = word;
            self.len += 1;
        }

        @memcpy(self.words[self.len..][0..trailing.len], trailing);
        self.len += trailing.len;
    }

    fn reserveIds(self: *Builder) Ids {
        return .{
            .main = self.nextId(),
            .wg = self.nextId(),
            .lid = self.nextId(),
            .a_var = self.nextId(),
            .b_var = self.nextId(),
            .c_var = self.nextId(),
            .pc_var = self.nextId(),

            .void_ty = self.nextId(),
            .bool_ty = self.nextId(),
            .u32_ty = self.nextId(),
            .i32_ty = self.nextId(),
            .f32_ty = self.nextId(),
            .in_ty = self.nextId(),
            .vec3_u32_ty = self.nextId(),
            .fn_void_ty = self.nextId(),

            .u32_0 = self.nextId(),
            .u32_1 = self.nextId(),
            .u32_2 = self.nextId(),
            .u32_3 = self.nextId(),
            .u32_8 = self.nextId(),
            .u32_16 = self.nextId(),
            .u32_24 = self.nextId(),
            .u32_32 = self.nextId(),
            .u32_64 = self.nextId(),
            .i32_0 = self.nextId(),
            .i32_2 = self.nextId(),
            .i32_3 = self.nextId(),
            .i32_4 = self.nextId(),
            .i32_5 = self.nextId(),
            .f32_0 = self.nextId(),

            .acc_ty = self.nextId(),
            .a_ty = self.nextId(),
            .b_ty = self.nextId(),
            .acc_zero = self.nextId(),

            .ptr_input_vec3_u32 = self.nextId(),
            .ptr_input_u32 = self.nextId(),
            .ptr_fn_u32 = self.nextId(),
            .ptr_fn_acc = self.nextId(),
            .ptr_pc_struct = self.nextId(),
            .ptr_pc_u32 = self.nextId(),
            .ptr_sb_f32 = self.nextId(),
            .ptr_sb_in = self.nextId(),

            .array_a = self.nextId(),
            .array_b = self.nextId(),
            .array_c = self.nextId(),
            .struct_a = self.nextId(),
            .struct_b = self.nextId(),
            .struct_c = self.nextId(),
            .ptr_sb_struct_a = self.nextId(),
            .ptr_sb_struct_b = self.nextId(),
            .ptr_sb_struct_c = self.nextId(),
            .pc_struct = self.nextId(),

            .acc_vars = .{ self.nextId(), self.nextId(), self.nextId(), self.nextId() },
            .kk_var = self.nextId(),
        };
    }
};

fn emitHeader(builder: *Builder) !void {
    if (builder.words.len < 5) return error.ModuleTooLarge;
    builder.words[0] = 0x07230203;
    builder.words[1] = 0x00010600;
    builder.words[2] = 0;
    builder.words[3] = 0;
    builder.words[4] = 0;
    builder.len = 5;
}

fn emitCapabilities(builder: *Builder, mode: Mode) !void {
    try builder.emit(OpCapability, &.{CapabilityShader});
    try builder.emit(OpCapability, &.{CapabilityStorageBuffer16BitAccess});
    try builder.emit(OpCapability, &.{CapabilityVulkanMemoryModel});
    try builder.emit(OpCapability, &.{CapabilityCooperativeMatrixKHR});
    if (mode.isF16()) {
        try builder.emit(OpCapability, &.{CapabilityFloat16});
    } else {
        try builder.emit(OpCapability, &.{CapabilityBFloat16TypeKHR});
        try builder.emit(OpCapability, &.{CapabilityBFloat16CooperativeMatrixKHR});
        try builder.emitString(OpExtension, &.{}, "SPV_KHR_bfloat16", &.{});
    }
    try builder.emitString(OpExtension, &.{}, "SPV_KHR_cooperative_matrix", &.{});
    try builder.emit(OpMemoryModel, &.{ AddressingModelLogical, MemoryModelVulkan });
}

fn emitEntry(builder: *Builder, ids: Ids, mode: Mode) !void {
    try builder.emitString(OpEntryPoint, &.{ ExecutionModelGLCompute, ids.main }, "main", &.{
        ids.wg,
        ids.lid,
        ids.a_var,
        ids.b_var,
        ids.c_var,
        ids.pc_var,
    });
    try builder.emit(OpExecutionMode, &.{ ids.main, ExecutionModeLocalSize, mode.localSizeX(), 1, 1 });
}

fn emitNames(builder: *Builder, ids: Ids, mode: Mode) !void {
    try builder.emitString(OpName, &.{ids.main}, "main", &.{});
    try builder.emitString(OpName, &.{ids.wg}, "gl_WorkGroupID", &.{});
    try builder.emitString(OpName, &.{ids.lid}, "gl_LocalInvocationID", &.{});
    try builder.emitString(OpName, &.{ids.a_var}, "a_buffer", &.{});
    try builder.emitString(OpName, &.{ids.b_var}, "b_buffer", &.{});
    try builder.emitString(OpName, &.{ids.c_var}, "c_buffer", &.{});
    try builder.emitString(OpName, &.{ids.pc_var}, "push_constants", &.{});
    try builder.emitString(OpName, &.{ids.struct_a}, "A", &.{});
    try builder.emitString(OpMemberName, &.{ ids.struct_a, 0 }, "data", &.{});
    try builder.emitString(OpName, &.{ids.struct_b}, "B", &.{});
    try builder.emitString(OpMemberName, &.{ ids.struct_b, 0 }, "data", &.{});
    try builder.emitString(OpName, &.{ids.struct_c}, "C", &.{});
    try builder.emitString(OpMemberName, &.{ ids.struct_c, 0 }, "data", &.{});
    try builder.emitString(OpName, &.{ids.pc_struct}, "PushConstants", &.{});
    try builder.emitString(OpName, &.{ids.in_ty}, mode.componentLabel(), &.{});
    if (mode.hasOptimizedNTiling()) try builder.emitString(OpName, &.{ids.acc_vars[0]}, "acc0", &.{});
}

fn emitDecorations(builder: *Builder, ids: Ids) !void {
    try builder.emit(OpDecorate, &.{ ids.wg, DecorationBuiltIn, BuiltInWorkgroupId });
    try builder.emit(OpDecorate, &.{ ids.lid, DecorationBuiltIn, 27 });

    try builder.emit(OpDecorate, &.{ ids.array_a, DecorationArrayStride, 2 });
    try builder.emit(OpDecorate, &.{ ids.array_b, DecorationArrayStride, 2 });
    try builder.emit(OpDecorate, &.{ ids.array_c, DecorationArrayStride, 4 });

    try builder.emit(OpDecorate, &.{ ids.struct_a, DecorationBlock });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_a, 0, DecorationNonWritable });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_a, 0, DecorationOffset, 0 });
    try builder.emit(OpDecorate, &.{ ids.a_var, DecorationNonWritable });
    try builder.emit(OpDecorate, &.{ ids.a_var, DecorationBinding, 0 });
    try builder.emit(OpDecorate, &.{ ids.a_var, DecorationDescriptorSet, 0 });

    try builder.emit(OpDecorate, &.{ ids.struct_b, DecorationBlock });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_b, 0, DecorationNonWritable });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_b, 0, DecorationOffset, 0 });
    try builder.emit(OpDecorate, &.{ ids.b_var, DecorationNonWritable });
    try builder.emit(OpDecorate, &.{ ids.b_var, DecorationBinding, 1 });
    try builder.emit(OpDecorate, &.{ ids.b_var, DecorationDescriptorSet, 0 });

    try builder.emit(OpDecorate, &.{ ids.struct_c, DecorationBlock });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_c, 0, DecorationNonReadable });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_c, 0, DecorationOffset, 0 });
    try builder.emit(OpDecorate, &.{ ids.c_var, DecorationNonReadable });
    try builder.emit(OpDecorate, &.{ ids.c_var, DecorationBinding, 2 });
    try builder.emit(OpDecorate, &.{ ids.c_var, DecorationDescriptorSet, 0 });

    try builder.emit(OpDecorate, &.{ ids.pc_struct, DecorationBlock });
    inline for (0..6) |i| {
        try builder.emit(OpMemberDecorate, &.{ ids.pc_struct, i, DecorationOffset, i * 4 });
    }
}

fn emitTypes(builder: *Builder, ids: Ids, mode: Mode) !void {
    const tile_n = u32IdForTileN(ids, mode.tileN());

    try builder.emit(OpTypeVoid, &.{ids.void_ty});
    try builder.emit(OpTypeBool, &.{ids.bool_ty});
    try builder.emit(OpTypeInt, &.{ ids.u32_ty, 32, 0 });
    try builder.emit(OpTypeInt, &.{ ids.i32_ty, 32, 1 });
    try builder.emit(OpTypeFloat, &.{ ids.f32_ty, 32 });
    if (mode.isF16()) {
        try builder.emit(OpTypeFloat, &.{ ids.in_ty, 16 });
    } else {
        try builder.emit(OpTypeFloat, &.{ ids.in_ty, 16, BFloat16EncodingKHR });
    }
    try builder.emit(OpTypeVector, &.{ ids.vec3_u32_ty, ids.u32_ty, 3 });
    try builder.emit(OpTypeFunction, &.{ ids.fn_void_ty, ids.void_ty });

    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_0, 0 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_1, 1 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_2, 2 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_3, 3 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_8, 8 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_16, 16 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_24, 24 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_32, 32 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_64, 64 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_0, 0 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_2, 2 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_3, 3 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_4, 4 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_5, 5 });
    try builder.emit(OpConstant, &.{ ids.f32_ty, ids.f32_0, 0 });

    try builder.emit(OpTypeCooperativeMatrixKHR, &.{ ids.acc_ty, ids.f32_ty, ids.u32_3, ids.u32_16, tile_n, ids.u32_2 });
    try builder.emit(OpTypeCooperativeMatrixKHR, &.{ ids.a_ty, ids.in_ty, ids.u32_3, ids.u32_16, ids.u32_16, ids.u32_0 });
    try builder.emit(OpTypeCooperativeMatrixKHR, &.{ ids.b_ty, ids.in_ty, ids.u32_3, ids.u32_16, tile_n, ids.u32_1 });
    try builder.emit(OpConstantComposite, &.{ ids.acc_ty, ids.acc_zero, ids.f32_0 });

    try builder.emit(OpTypePointer, &.{ ids.ptr_input_vec3_u32, StorageClassInput, ids.vec3_u32_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_input_u32, StorageClassInput, ids.u32_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_fn_u32, StorageClassFunction, ids.u32_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_fn_acc, StorageClassFunction, ids.acc_ty });

    try builder.emit(OpTypeRuntimeArray, &.{ ids.array_a, ids.in_ty });
    try builder.emit(OpTypeRuntimeArray, &.{ ids.array_b, ids.in_ty });
    try builder.emit(OpTypeRuntimeArray, &.{ ids.array_c, ids.f32_ty });
    try builder.emit(OpTypeStruct, &.{ ids.struct_a, ids.array_a });
    try builder.emit(OpTypeStruct, &.{ ids.struct_b, ids.array_b });
    try builder.emit(OpTypeStruct, &.{ ids.struct_c, ids.array_c });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_struct_a, StorageClassStorageBuffer, ids.struct_a });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_struct_b, StorageClassStorageBuffer, ids.struct_b });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_struct_c, StorageClassStorageBuffer, ids.struct_c });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_in, StorageClassStorageBuffer, ids.in_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_f32, StorageClassStorageBuffer, ids.f32_ty });

    try builder.emit(OpTypeStruct, &.{ ids.pc_struct, ids.u32_ty, ids.u32_ty, ids.u32_ty, ids.u32_ty, ids.u32_ty, ids.u32_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_pc_struct, StorageClassPushConstant, ids.pc_struct });
    try builder.emit(OpTypePointer, &.{ ids.ptr_pc_u32, StorageClassPushConstant, ids.u32_ty });

    try builder.emit(OpVariable, &.{ ids.ptr_input_vec3_u32, ids.wg, StorageClassInput });
    try builder.emit(OpVariable, &.{ ids.ptr_input_vec3_u32, ids.lid, StorageClassInput });
    try builder.emit(OpVariable, &.{ ids.ptr_sb_struct_a, ids.a_var, StorageClassStorageBuffer });
    try builder.emit(OpVariable, &.{ ids.ptr_sb_struct_b, ids.b_var, StorageClassStorageBuffer });
    try builder.emit(OpVariable, &.{ ids.ptr_sb_struct_c, ids.c_var, StorageClassStorageBuffer });
    try builder.emit(OpVariable, &.{ ids.ptr_pc_struct, ids.pc_var, StorageClassPushConstant });
}

fn accessChain(builder: *Builder, result_ty: u32, base: u32, indexes: []const u32) !u32 {
    const result = builder.nextId();
    var operands: [8]u32 = undefined;
    if (indexes.len + 3 > operands.len) return error.TooManyIndexes;
    operands[0] = result_ty;
    operands[1] = result;
    operands[2] = base;
    @memcpy(operands[3..][0..indexes.len], indexes);
    try builder.emit(OpAccessChain, operands[0 .. indexes.len + 3]);
    return result;
}

fn load(builder: *Builder, result_ty: u32, ptr: u32) !u32 {
    const result = builder.nextId();
    try builder.emit(OpLoad, &.{ result_ty, result, ptr });
    return result;
}

fn binary(builder: *Builder, opcode: u16, result_ty: u32, lhs: u32, rhs: u32) !u32 {
    const result = builder.nextId();
    try builder.emit(opcode, &.{ result_ty, result, lhs, rhs });
    return result;
}

fn coopLoad(builder: *Builder, result_ty: u32, ptr: u32, stride: u32, row_major: u32) !u32 {
    const result = builder.nextId();
    try builder.emit(OpCooperativeMatrixLoadKHR, &.{ result_ty, result, ptr, row_major, stride, MemoryAccessNone });
    return result;
}

fn renderZigWords(gpa: std.mem.Allocator, shader: CoopMatmulShader, words: []const u32) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    const writer = &out.writer;
    defer out.deinit();

    try writer.print(
        \\//! Generated by `//tools:emit_coop_dsl_spv`.
        \\//! Do not edit by hand.
        \\
        \\pub const import_name = "{s}";
        \\pub const words = [_]u32{{
        \\
    , .{shader.import_name});

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
