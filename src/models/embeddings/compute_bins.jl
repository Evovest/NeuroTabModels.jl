"""
    compute_bins(X; bins=48)

Compute quantile-based bin boundaries for `PiecewiseLinearEncoding`/`PiecewiseLinearEmbeddings`.

Note: `X` should have shape `(n_samples, n_features)`, which is the transpose of the
model input convention `(n_features, batch)`. Transpose your data before calling this.

# Arguments
- `X::AbstractMatrix`: Training data of shape `(n_samples, n_features)`.
- `bins::Union{Int, Vector{Int}}`: Number of bins per feature (default `48`).
  A single `Int` applies the same count to all features. A `Vector{Int}` of length
  `n_features` specifies per-feature bin counts.

# Returns
- `Vector{Vector{Float32}}`: Bin edges for each feature. Each vector has between 2 and
  `bins + 1` elements (fewer if quantiles coincide for low-cardinality features).
"""
function compute_bins(X::AbstractMatrix; bins::Union{Int,Vector{Int}}=48)
    n_samples, n_features = size(X)

    n_bins_vec = if bins isa Int
        @assert bins > 1 "bins must be > 1, got $bins"
        @assert bins < n_samples "bins must be < n_samples, got bins=$bins, n_samples=$n_samples"
        fill(bins, n_features)
    else
        @assert length(bins) == n_features "Length of bins must match n_features ($n_features), got $(length(bins))"
        for (j, nb) in enumerate(bins)
            @assert nb > 1 "bins[$j] must be > 1, got $nb"
            @assert nb < n_samples "bins[$j] must be < n_samples, got bins=$nb, n_samples=$n_samples"
        end
        bins
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