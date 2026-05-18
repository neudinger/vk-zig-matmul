def _zig_shader_spv_words_impl(ctx):
    spvasm = ctx.actions.declare_file(ctx.label.name + ".spvasm")
    spv = ctx.actions.declare_file(ctx.label.name + ".spv")
    zig_words = ctx.actions.declare_file(ctx.label.name + ".zig")

    frontend_args = ctx.actions.args()
    frontend_args.add("--mode=" + ctx.attr.mode)
    frontend_args.add(ctx.file.src)
    frontend_args.add(spvasm)
    ctx.actions.run(
        executable = ctx.executable._frontend,
        arguments = [frontend_args],
        inputs = [ctx.file.src],
        outputs = [spvasm],
        mnemonic = "ZigShaderSpvasm",
        progress_message = "Emitting SPIR-V assembly %{output}",
    )

    as_args = ctx.actions.args()
    as_args.add("--target-env")
    as_args.add(ctx.attr.target_env)
    as_args.add(spvasm)
    as_args.add("-o")
    as_args.add(spv)
    ctx.actions.run(
        executable = ctx.executable._spirv_as,
        arguments = [as_args],
        inputs = [spvasm],
        tools = [ctx.executable._spirv_as],
        outputs = [spv],
        mnemonic = "SpirvAssemble",
        progress_message = "Assembling SPIR-V %{output}",
    )

    embed_args = ctx.actions.args()
    embed_args.add("--spirv-val")
    embed_args.add(ctx.executable._spirv_val)
    embed_args.add("--target-env")
    embed_args.add(ctx.attr.target_env)
    embed_args.add(spv)
    embed_args.add(zig_words)
    embed_args.add(ctx.attr.import_name)
    ctx.actions.run(
        executable = ctx.executable._validate_embed,
        arguments = [embed_args],
        inputs = [spv],
        tools = [
            ctx.executable._spirv_val,
            ctx.executable._validate_embed,
        ],
        outputs = [zig_words],
        mnemonic = "SpirvValidateEmbed",
        progress_message = "Validating and embedding SPIR-V %{output}",
    )

    return [
        DefaultInfo(files = depset([zig_words])),
        OutputGroupInfo(
            spvasm = depset([spvasm]),
            spv = depset([spv]),
            all_shader_artifacts = depset([spvasm, spv, zig_words]),
        ),
    ]

zig_shader_spv_words = rule(
    implementation = _zig_shader_spv_words_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".zig"],
            mandatory = True,
        ),
        "mode": attr.string(mandatory = True),
        "target_env": attr.string(default = "vulkan1.4"),
        "import_name": attr.string(mandatory = True),
        "_frontend": attr.label(
            default = Label("//tools:emit_zig_shader_spvasm"),
            executable = True,
            cfg = "exec",
        ),
        "_validate_embed": attr.label(
            default = Label("//tools:validate_embed_spv"),
            executable = True,
            cfg = "exec",
        ),
        "_spirv_as": attr.label(
            default = Label("@spirv_tools//:spirv-as"),
            executable = True,
            cfg = "exec",
        ),
        "_spirv_val": attr.label(
            default = Label("@spirv_tools//:spirv-val"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def _slang_shader_spv_words_impl(ctx):
    raw_spvasm = ctx.actions.declare_file(ctx.label.name + ".raw.spvasm")
    spvasm = ctx.actions.declare_file(ctx.label.name + ".spvasm")
    spv = ctx.actions.declare_file(ctx.label.name + ".spv")
    zig_words = ctx.actions.declare_file(ctx.label.name + ".zig")

    slang_args = ctx.actions.args()
    slang_args.add(ctx.file.src)
    slang_args.add("-target")
    slang_args.add("spirv-asm")
    slang_args.add("-entry")
    slang_args.add(ctx.attr.entry)
    slang_args.add("-stage")
    slang_args.add(ctx.attr.stage)
    slang_args.add("-capability")
    slang_args.add(ctx.attr.capabilities)
    slang_args.add(ctx.attr.optimization)
    slang_args.add("-o")
    slang_args.add(raw_spvasm)
    ctx.actions.run(
        executable = ctx.executable._slangc,
        arguments = [slang_args],
        inputs = [ctx.file.src],
        tools = ctx.files._slang_runtime,
        outputs = [raw_spvasm],
        mnemonic = "SlangShaderSpvasm",
        progress_message = "Compiling Slang shader %{input} to SPIR-V assembly",
    )

    ctx.actions.run_shell(
        arguments = [raw_spvasm.path, spvasm.path],
        command = """sed 's/^OpCapability UniformAndStorageBuffer16BitAccess$/OpCapability StorageBuffer16BitAccess/' "$1" > "$2" """,
        inputs = [raw_spvasm],
        outputs = [spvasm],
        mnemonic = "NormalizeSlangSpvasm",
        progress_message = "Normalizing Slang SPIR-V assembly %{output}",
    )

    as_args = ctx.actions.args()
    as_args.add("--target-env")
    as_args.add(ctx.attr.target_env)
    as_args.add(spvasm)
    as_args.add("-o")
    as_args.add(spv)
    ctx.actions.run(
        executable = ctx.executable._spirv_as,
        arguments = [as_args],
        inputs = [spvasm],
        tools = [ctx.executable._spirv_as],
        outputs = [spv],
        mnemonic = "SpirvAssemble",
        progress_message = "Assembling SPIR-V %{output}",
    )

    embed_args = ctx.actions.args()
    embed_args.add("--spirv-val")
    embed_args.add(ctx.executable._spirv_val)
    embed_args.add("--target-env")
    embed_args.add(ctx.attr.target_env)
    embed_args.add(spv)
    embed_args.add(zig_words)
    embed_args.add(ctx.attr.import_name)
    ctx.actions.run(
        executable = ctx.executable._validate_embed,
        arguments = [embed_args],
        inputs = [spv],
        tools = [
            ctx.executable._spirv_val,
            ctx.executable._validate_embed,
        ],
        outputs = [zig_words],
        mnemonic = "SpirvValidateEmbed",
        progress_message = "Validating and embedding SPIR-V %{output}",
    )

    return [
        DefaultInfo(files = depset([zig_words])),
        OutputGroupInfo(
            raw_spvasm = depset([raw_spvasm]),
            spvasm = depset([spvasm]),
            spv = depset([spv]),
            all_shader_artifacts = depset([raw_spvasm, spvasm, spv, zig_words]),
        ),
    ]

slang_shader_spv_words = rule(
    implementation = _slang_shader_spv_words_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".slang"],
            mandatory = True,
        ),
        "entry": attr.string(default = "main"),
        "stage": attr.string(default = "compute"),
        "capabilities": attr.string(default = "cooperative_matrix_tensor_addressing+tensor_addressing+cooperative_matrix_2"),
        "optimization": attr.string(default = "-O3"),
        "target_env": attr.string(default = "vulkan1.4"),
        "import_name": attr.string(mandatory = True),
        "_slangc": attr.label(
            default = Label("@slang//:bin/slangc"),
            executable = True,
            cfg = "exec",
            allow_single_file = True,
        ),
        "_slang_runtime": attr.label(
            default = Label("@slang//:slang_runtime"),
            cfg = "exec",
        ),
        "_validate_embed": attr.label(
            default = Label("//tools:validate_embed_spv"),
            executable = True,
            cfg = "exec",
        ),
        "_spirv_as": attr.label(
            default = Label("@spirv_tools//:spirv-as"),
            executable = True,
            cfg = "exec",
        ),
        "_spirv_val": attr.label(
            default = Label("@spirv_tools//:spirv-val"),
            executable = True,
            cfg = "exec",
        ),
    },
)
