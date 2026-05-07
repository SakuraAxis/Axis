function normalize2(x::Float32, y::Float32)
    len = sqrt(x * x + y * y)
    return len < 1f-6 ? (0f0, 0f0) : (x / len, y / len)
end

normalize2(x::Real, y::Real) = normalize2(Float32(x), Float32(y))
