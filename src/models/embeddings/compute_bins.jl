"""
    compute_bins(X; n_bins=48)

Compute quantile-based bin boundaries for `PiecewiseLinearEncoding`/`PiecewiseLinearEmbeddings`.

Note: `X` should have shape `(n_samples, n_features)`, which is the transpose of the
model input convention `(n_features, batch)`. Transpose your data before calling this.

# Arguments
- `X::AbstractMatrix`: Training data of shape `(n_samples, n_features)`.
- `n_bins::Union{Int, Vector{Int}}`: Number of bins per feature (default `48`).
  A single `Int` applies the same count to all features. A `Vector{Int}` of length
  `n_features` specifies per-feature bin counts.

# Returns
- `Vector{Vector{Float32}}`: Bin edges for each feature. Each vector has between 2 and
  `n_bins + 1` elements (fewer if quantiles coincide for low-cardinality features).
"""
function compute_bins(X::AbstractMatrix; n_bins::Union{Int, Vector{Int}}=48)
    n_samples, n_features = size(X)

    n_bins_vec = if n_bins isa Int
        @assert n_bins > 1 "n_bins must be > 1, got $n_bins"
        @assert n_bins < n_samples "n_bins must be < n_samples, got n_bins=$n_bins, n_samples=$n_samples"
        fill(n_bins, n_features)
    else
        @assert length(n_bins) == n_features "Length of n_bins must match n_features ($n_features), got $(length(n_bins))"
        for (j, nb) in enumerate(n_bins)
            @assert nb > 1 "n_bins[$j] must be > 1, got $nb"
            @assert nb < n_samples "n_bins[$j] must be < n_samples, got n_bins=$nb, n_samples=$n_samples"
        end
        n_bins
    end

    bins = Vector{Vector{Float32}}(undef, n_features)
    col_buf = Vector{eltype(X)}(undef, n_samples)

    for j in 1:n_features
        copyto!(col_buf, view(X, :, j))
        sort!(col_buf)
        quantile_probs = range(0f0, 1f0, length=n_bins_vec[j] + 1)
        edges = Float32[quantile(col_buf, p; sorted=true) for p in quantile_probs]
        unique!(edges)
        @assert length(edges) >= 2 "Feature $j has fewer than 2 unique bin edges"
        bins[j] = edges
    end
    return bins
end