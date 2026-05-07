using Libdl

const _LIB_NAME = Sys.iswindows() ? "axis_rs.dll" :
                  Sys.isapple() ? "libaxis_rs.dylib" :
                  "libaxis_rs.so"
const _AXIS_RS_LIB = Ref{String}("")
const _AXIS_RS_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _AXIS_RS_SYMBOLS = IdDict{Symbol, Ptr{Cvoid}}()

function _default_axis_rs_library_path()
    package_dir = dirname(@__DIR__)
    workspace_dir = dirname(package_dir)
    return joinpath(workspace_dir, "axis_rs", "target", "release", _LIB_NAME)
end

function _resolve_axis_rs_library_path()
    env_path = get(ENV, "AXIS_RS_LIB", "")
    isempty(env_path) || return env_path
    return _default_axis_rs_library_path()
end

function _init_axis_rs!()
    empty!(_AXIS_RS_SYMBOLS)
    _AXIS_RS_HANDLE[] = C_NULL

    lib_path = _resolve_axis_rs_library_path()
    _AXIS_RS_LIB[] = lib_path

    if isfile(lib_path)
        _AXIS_RS_HANDLE[] = Libdl.dlopen(lib_path)
    end

    return nothing
end

"""
    axis_rs_library_path()

Return the Rust shared library path used by this package.
Set `ENV["AXIS_RS_LIB"]` before loading `Axis` to override it.
"""
axis_rs_library_path() = _AXIS_RS_LIB[]

"""
    axis_rs_available()

Return `true` when the Rust shared library exists at `axis_rs_library_path()`.
"""
axis_rs_available() = isfile(_AXIS_RS_LIB[]) && _AXIS_RS_HANDLE[] != C_NULL

function _check_axis_rs_available()
    axis_rs_available() && return nothing
    error(
        "Axis Rust library was not found at $(_AXIS_RS_LIB[]). " *
        "Build it with `cargo build --release` in the `axis_rs` directory, " *
        "or set `ENV[\"AXIS_RS_LIB\"]` before `using Axis`."
    )
end

function _axis_rs_symbol(name::Symbol)
    _check_axis_rs_available()
    return get!(_AXIS_RS_SYMBOLS, name) do
        Libdl.dlsym(_AXIS_RS_HANDLE[], name)
    end
end
