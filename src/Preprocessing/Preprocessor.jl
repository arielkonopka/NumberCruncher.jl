module Preprocessor
using Statistics

export TRANSFORMS, prepare_series

"""
Available preparation steps:
- :raw     - the values as they are (e.g. a Lorenz coordinate)
- :diff    - first differences v[i+1]-v[i] (what deltaO is for prices)
- :logdiff - log returns log(v[i+1])-log(v[i]); needs positive values
"""
const TRANSFORMS = (:raw, :diff, :logdiff)

"""
Turns a raw column into the series the estimator works on.

Parameters:
- values: raw column, may contain `missing`/NaN (those rows are skipped)
- labels: one label per raw row (time stamps, row numbers, ...)
- transform: one of TRANSFORMS
- noise: Gaussian noise added after the transform, as a fraction of the series' std

Returns:
- (series, labels): series::Vector{Float64} and the label of each series element;
  a difference is labelled with the row at which it ends
"""
function prepare_series(values::AbstractVector, labels::AbstractVector=1:length(values);
                        transform::Symbol=:diff, noise::Real=0.0)
    transform in TRANSFORMS || throw(ArgumentError("unknown transform $transform; use one of $TRANSFORMS"))
    keep = [!ismissing(v) && v isa Real && isfinite(v) for v in values]
    v = Float64.(values[keep])
    l = collect(labels)[keep]
    length(v) < 3 && throw(ArgumentError("column has fewer than 3 usable values"))

    if transform == :raw
        s = v
    elseif transform == :diff
        s, l = diff(v), l[2:end]
    else
        any(<=(0), v) && throw(ArgumentError("log returns need strictly positive values"))
        s, l = diff(log.(v)), l[2:end]
    end

    if noise > 0
        s = s .+ (noise * std(s)) .* randn(length(s))
    end
    return (series=s, labels=l)
end

end
