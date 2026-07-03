"""
    compute_bins(X; nbins=48)

Quantile-based bin edges for piecewise-linear embeddings.

`X` must have shape `(n_samples, n_features)`,the transpose of model input `(n_features, batch)`.

# Arguments
- `X::AbstractMatrix`: Training data `(n_samples, n_features)`.
- `bins::Union{Int, Vector{Int}}`: Bin count per feature (default `32`).

# Returns
`Vector{Vector{Float32}}` of bin edges per feature.
"""
function compute_bins(X::AbstractMatrix; nbins::Int=32)
    n_samples, n_features = size(X)

    @assert nbins > 1 "bins must be > 1, got $nbins"
    @assert nbins < n_samples "bins must be < n_samples, got bins=$nbins, n_samples=$n_samples"

    bins = Vector{Vector{Float32}}(undef, n_features)
    col_buf = Vector{eltype(X)}(undef, n_samples)

    for j in 1:n_features
        copyto!(col_buf, view(X, :, j))
        sort!(col_buf)
        quantile_probs = range(0f0, 1f0, length=nbins + 1)
        edges = Float32[quantile(col_buf, p; sorted=true) for p in quantile_probs]
        unique!(edges)
        if length(edges) < 2
            # Constant features collapse all quantile edges to one value.
            # Give the encoder a tiny nonzero-width bin instead of failing.
            v = edges[1]
            delta = max(abs(v) * 1f-3, 1f-3)
            edges = Float32[v-delta, v+delta]
        end
        bins[j] = edges
    end
    return bins
end
