"""
    compute_bins(X; bins=32)

Quantile-based bin edges for piecewise-linear embeddings.

`X` must have shape `(n_samples, n_features)`,the transpose of model input `(n_features, batch)`.

# Arguments
- `X::AbstractMatrix`: Training data `(n_samples, n_features)`.
- `bins::Int`: Bin count per feature (default `32`).

# Returns
`Vector{Vector{Float32}}` of bin edges per feature.
"""
function compute_bins(X::AbstractMatrix; bins::Int=32)
    n_samples, n_features = size(X)

    @assert bins > 1 "bins must be > 1, got $bins"
    @assert bins < n_samples "bins must be < n_samples, got bins=$bins, n_samples=$n_samples"

    edges = Vector{Vector{Float32}}(undef, n_features)
    col_buf = Vector{eltype(X)}(undef, n_samples)

    for j in 1:n_features
        copyto!(col_buf, view(X, :, j))
        sort!(col_buf)
        quantile_probs = range(0.0f0, 1.0f0; length=bins + 1)
        feat_edges = Float32[quantile(col_buf, p; sorted=true) for p in quantile_probs]
        unique!(feat_edges)
        if length(feat_edges) < 2
            # Constant features collapse all quantile edges to one value.
            # Give the encoder a tiny nonzero-width bin instead of failing.
            v = feat_edges[1]
            delta = max(abs(v) * 1.0f-3, 1.0f-3)
            feat_edges = Float32[v - delta, v + delta]
        end
        edges[j] = feat_edges
    end
    return edges
end
