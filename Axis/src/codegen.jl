
#=
Axis Code Generator

Provides two macros and one generator function :

  @rust_fn  function name(args...)::RetType
      """
      // Rust body here
      """
  end

  @call_rust_fn name(args...)

  @rust_code """
  // Arbitrary Rust (structs, use statements, helpers, ...)
  """

  AX.generate_bridge(axis_rs_path)   # writes generated/*.rs + julia bindings

Files are grouped by the Julia source file that called the macro.
  PhillipsOceanAX.jl  ->  axis_rs/src/generated/phillips_ocean_ax.rs
                           Achernar/src/generated_bindings/phillips_ocean_ax_bindings.jl
=#

#=
AST helpers
=#

function _parse_fn_expr(expr::Expr)
    expr.head == :function ||
        error("@rust_fn: expected `function ... end` expression, got `$(expr.head)`.")

    sig_part = expr.args[1]
    body_part = expr.args[2] # Expr(:block, ...)

    # Return type annotation?  function foo(...)::T  vs  function foo(...)
    if sig_part isa Expr && sig_part.head == :(::)
        ret_jl   = sig_part.args[2]
        call_expr = sig_part.args[1]
    else
        ret_jl   = :Cvoid
        call_expr = sig_part
    end

    call_expr isa Expr && call_expr.head == :call ||
        error("@rust_fn: cannot parse function call signature.")

    fn_name = call_expr.args[1] # Symbol, e.g. :_pack_components!

    params = []
    for p in call_expr.args[2:end]
        if p isa Expr && p.head == :(::)
            push!(params, (name = p.args[1], jl_type = p.args[2]))
        else
            error("@rust_fn: every parameter must have an explicit type annotation, got `$p`.")
        end
    end

    # Extract the Rust body string from the function block.
    # We expect exactly one String or interpolated string literal.
    rust_body = nothing
    for stmt in body_part.args
        stmt isa LineNumberNode && continue
        if stmt isa String
            rust_body = stmt
            break
        end
        if stmt isa Expr && stmt.head == :string
            rust_body = join(string.(stmt.args))
            break
        end
    end
    rust_body !== nothing ||
        error("@rust_fn `$fn_name`: body must be a string literal containing Rust code.")

    return fn_name, params, ret_jl, rust_body
end

#=
@rust_fn  macro
=#

"""
    @rust_fn function name(arg::JuliaType, ...)::ReturnType
        \"\"\"
        // Rust function body
        \"\"\"
    end

Declare a Rust FFI function.  Axis will :
  1. Register the Rust source keyed to the current Julia source file.
  2. Generate a private `ccall` Julia stub with the same signature,
     zero overhead when called via `@call_rust_fn`.
  3. Emit `#[unsafe(no_mangle)] pub unsafe extern "C" fn ...` in the
     corresponding `axis_rs/src/generated/<mod>.rs` when you call
     `AX.generate_bridge(path)`.

Supported parameter types: Float32, Float64, Int32, Int64, UInt8, UInt16,
UInt32, UInt64, Bool, Cvoid, Csize_t, Cint, Cfloat, Cdouble, Ptr{T}.
"""
macro rust_fn(expr)
    fn_name, params, ret_jl, rust_body = _parse_fn_expr(expr)

    rust_name  = _rust_fn_name(fn_name)
    rust_ret   = _jl_to_rust(ret_jl)
    stub_name  = _rust_stub_name(fn_name)

    # Build Rust parameter list: "name: rust_type"
    rust_params = join(
        ["$(p.name): $(_jl_to_rust(p.jl_type))" for p in params],
        ", "
    )

    # Build Rust source for this function
    rust_src = """
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn $rust_name($rust_params) -> $rust_ret {
    $(rust_body)
    }
    """

    # Source file info captured at macro-expansion time
    src_file = string(__source__.file)
    jl_file = basename(src_file)
    rust_mod = _jl_filename_to_rust_mod(src_file)
    mod_info = (name = rust_mod, orig_file = jl_file)

    # Precompilation-safe registration via method definition on Axis.register_rust_fn
    registration = :(Axis.register_rust_fn(::Val{$(Expr(:quote, Symbol(rust_name)))}) = (
        rust_name = $rust_name, 
        rust_src = $rust_src, 
        rust_mod = $rust_mod,
        mod_info = $mod_info
    ))

    # Build escaped parameter lists, argument lists and type tuple
    jl_params = [Expr(:(::), esc(p.name), esc(p.jl_type)) for p in params]
    ccall_types = Expr(:tuple, [esc(p.jl_type) for p in params]...)
    ccall_args = [esc(p.name) for p in params]

    # Explicitly construct the function body with __source__ for correct line numbers
    fn_body = Expr(:block,
        __source__,
        Expr(:call, :ccall,
            Expr(:call, :(Axis._axis_rs_symbol), Expr(:quote, Symbol(rust_name))),
            esc(ret_jl),
            ccall_types,
            ccall_args...
        )
    )

    # Explicitly construct the function definition Expr
    fn_expr = Expr(:function,
        Expr(:call, esc(stub_name), jl_params...),
        fn_body
    )

    mod_registration = :(Axis.register_module_info(::Val{$(Expr(:quote, Symbol(rust_mod)))}) = $mod_info)

    # Return a flat Expr(:block) so that Core.@doc can inspect and document fn_expr correctly
    return Expr(:block,
        __source__,
        registration,
        mod_registration,
        fn_expr
    )
end

#=
@call_rust_fn macro
=#

"""
    @call_rust_fn name(args...)

Call a Rust function declared with `@rust_fn`. This keeps Rust-backed calls
visibly distinct from ordinary Julia calls while still compiling to the same
direct `ccall` stub.
"""
macro call_rust_fn(expr)
    expr isa Expr && expr.head == :call ||
        error("@call_rust_fn: expected a function call expression, got `$expr`.")

    fn_name = expr.args[1]
    fn_name isa Symbol ||
        error("@call_rust_fn: expected an unqualified function name, got `$fn_name`.")

    stub_name = _rust_stub_name(fn_name)
    args = [esc(arg) for arg in expr.args[2:end]]

    return Expr(:block,
        __source__,
        Expr(:call, esc(stub_name), args...)
    )
end

#=
@rust_code  macro
=#

"""
    @rust_code \"\"\"
    // Any valid Rust — structs, use statements, helper functions, ...
    \"\"\"

Inject raw Rust source into the generated module file that corresponds to
the current Julia source file.  Emitted *before* the `@rust_fn` functions,
so you can define types and helpers here and use them in `@rust_fn` bodies.
"""
macro rust_code(code_str)
    code_str isa String ||
        error("@rust_code: argument must be a string literal.")

    src_file = string(__source__.file)
    jl_file = basename(src_file)
    rust_mod = _jl_filename_to_rust_mod(src_file)
    mod_info = (name = rust_mod, orig_file = jl_file)

    # Precompilation-safe unique ID generation at macro-expansion time
    unique_id = hash(code_str) ⊻ rand(UInt64)

    return quote
        Axis.register_rust_code(::Val{$(Expr(:quote, Symbol(rust_mod)))}, ::Val{$unique_id}) = $code_str
        Axis.register_module_info(::Val{$(Expr(:quote, Symbol(rust_mod)))}) = $mod_info
    end
end

#=
generate_bridge
=#

"""
    generate_bridge(axis_rs_path; bindings_path=nothing)

Write all registered `@rust_fn` / `@rust_code` declarations into the
`axis_rs/src/generated/` directory, one `.rs` file per originating Julia
source file.  Also writes the corresponding Julia `ccall` binding files.

# Arguments
- `axis_rs_path`  : absolute path to the `axis_rs` Rust crate root.
- `bindings_path` : where to write the Julia binding `.jl` files.
                    Defaults to `axis_rs_path/../<caller_package>/src/generated_bindings/`.
                    Usually you will pass this explicitly.

# Workflow
1. Call `generate_bridge(...)` from your Julia session.
2. Open VS Code / your Rust IDE — inspect the generated `.rs` files.
3. Add any needed crate dependencies with `cargo add` in `axis_rs/`.
4. Run `cargo build --release` in `axis_rs/`.
5. `using Axis` (or restart your session) to load the freshly compiled lib.
"""
function generate_bridge(output_dir::String; bindings_path::Union{String,Nothing}=nothing)
    mkpath(output_dir)
 
    fns_by_mod = Dict{String, Vector{NamedTuple}}()
    code_by_mod = Dict{String, Vector{String}}()
    known_rust_fn_names = Dict{String, String}() # name => mod

    modules_metadata = Dict{String, NamedTuple}()

    # 1. Retrieve the uppercase original filenames of all modules from the dedicated metadata table
    for m in methods(register_module_info)
        if length(m.sig.parameters) == 2
            val_type = m.sig.parameters[2]
            if val_type <: Val
                mod_sym = val_type.parameters[1]
                modules_metadata[string(mod_sym)] = register_module_info(Val(mod_sym))
            end
        end
    end
 
    # 2. Gather all @rust_fn registrations from register_rust_fn method table
    for m in methods(register_rust_fn)
        if length(m.sig.parameters) == 2
            val_type = m.sig.parameters[2]
            if val_type <: Val
                rust_sym = val_type.parameters[1]
                entry = register_rust_fn(Val(rust_sym))

                # Check for duplicate names
                if haskey(known_rust_fn_names, entry.rust_name)
                    existing = known_rust_fn_names[entry.rust_name]
                    if existing != entry.rust_mod
                        error("Axis @rust_fn: duplicate Rust function name `$(entry.rust_name)` " *
                              "defined in both `$existing` and `$(entry.rust_mod)`. " *
                              "Rust C symbols are global — each name must be unique.")
                    end
                else
                    known_rust_fn_names[entry.rust_name] = entry.rust_mod
                    bucket = get!(fns_by_mod, entry.rust_mod, NamedTuple[])
                    push!(bucket, (rust_name = entry.rust_name, rust_src = entry.rust_src))
                end
            end
        end
    end
 
    # 3. Gather all @rust_code registrations from register_rust_code method table
    for m in methods(register_rust_code)
        if length(m.sig.parameters) == 3
            mod_type = m.sig.parameters[2]
            id_type = m.sig.parameters[3]
            if mod_type <: Val && id_type <: Val
                mod_sym = mod_type.parameters[1]
                id_val = id_type.parameters[1]
                
                rust_mod = string(mod_sym)
                code_str = register_rust_code(Val(mod_sym), Val(id_val))
                
                bucket = get!(code_by_mod, rust_mod, String[])
                push!(bucket, code_str)
            end
        end
    end
 
    all_mods = union(keys(fns_by_mod), keys(code_by_mod))
 
    isempty(all_mods) &&
        @warn "Axis.generate_bridge: no @rust_fn or @rust_code declarations found. " *
              "Make sure your modules are loaded before calling generate_bridge."
 
    #=
    Write one .rs file per Julia source file
    =#
    for mod_name in sort(collect(all_mods))
        rs_path = joinpath(output_dir, "$mod_name.rs")

        meta = get(modules_metadata, mod_name, (name = mod_name, orig_file = "$(mod_name).jl"))
 
        open(rs_path, "w") do io
            println(io, "// AUTO-GENERATED by Axis.bridge_up() — DO NOT EDIT MANUALLY.")
            println(io, "// Source Julia file : $(meta.orig_file)")
            println(io, "// Re-generate with : AX.bridge_up(\"path/to/axis_rs\")")
            println(io)
 
            # Raw code blocks first (use statements, structs, helpers)
            for raw in get(code_by_mod, mod_name, String[])
                println(io, raw)
                println(io)
            end
 
            # FFI functions
            for entry in get(fns_by_mod, mod_name, NamedTuple[])
                println(io, entry.rust_src)
            end
        end
 
        @info "Axis: wrote Rust module" file=rs_path
    end
 
    #=
    Write mod.rs ( declares all sub-modules )
    =#
    mod_rs_path = joinpath(output_dir, "mod.rs")
    open(mod_rs_path, "w") do io
        println(io, "// AUTO-GENERATED by Axis.bridge_up() — DO NOT EDIT MANUALLY.")
        println(io)
        for mod_name in sort(collect(all_mods))
            println(io, "pub mod $mod_name;")
        end
    end
    @info "Axis : wrote mod.rs" file=mod_rs_path
 
    #=
    Write Julia binding files
    =#
    if bindings_path !== nothing
        mkpath(bindings_path)
        for mod_name in sort(collect(all_mods))
            fns = get(fns_by_mod, mod_name, NamedTuple[])
            isempty(fns) && continue
 
            jl_path = joinpath(bindings_path, "$(mod_name)_bindings.jl")
            open(jl_path, "w") do io
                println(io, "# AUTO-GENERATED by Axis.bridge_up() — DO NOT EDIT MANUALLY.")
                println(io, "# Source module: $mod_name")
                println(io)
                println(io, "# Private ccall stubs are injected into your module by @rust_fn.")
                println(io, "# Call them with @AX.call_rust_fn name(args...).")
                println(io, "# This file is for reference / manual inclusion only.")
                println(io)
                for entry in fns
                    println(io, "# $(entry.rust_name)")
                    println(io)
                end
            end
            @info "Axis: wrote Julia bindings" file=jl_path
        end
    end
 
    @info "Axis.generate_bridge complete. Next steps:" *
          "\n  1. Inspect the generated Rust files in: $output_dir" *
          "\n  2. Copy/integrate them into your Rust workspace as needed"
 
    return nothing
end
