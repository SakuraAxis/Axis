module Axis

include("ffi.jl")
include("math/vector.jl")
include("ocean/phillips_ocean.jl")

export GRAVITY, WIND_SPEED
export normalize2, phillips_spectrum
export axis_rs_library_path, axis_rs_available

function __init__()
    _init_axis_rs!()
    return nothing
end

end # module Axis
