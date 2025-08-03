"""
    leaf_weights!(cw, nw)

Compute the cumulative probability associated with each node, down to terminal leaves
"""
function leaf_weights!(nw)
    cw = ones(eltype(nw), 2 * size(nw, 1) + 1, size(nw)[2:3]...)
    @threads for batch in axes(nw, 3)
        @inbounds for tree in axes(nw, 2)
            for i = 2:2:size(cw, 1)
                # child cumulative probability is obtained from the product of the parent cumulative prob (cw[parent]) and the parent probability (np[parent])
                cw[i, tree, batch] = cw[i>>1, tree, batch] * nw[i>>1, tree, batch]
                cw[i+1, tree, batch] = cw[i>>1, tree, batch] * (1 - nw[i>>1, tree, batch])
            end
        end
    end
    @views lw = cw[size(nw, 1)+1:size(cw, 1), :, 1:size(nw, 3)]
    return (cw, lw)
end

"""
    leaf_weights!(cw::AnyCuArray, nw::AnyCuArray)

Compute the cumulative probability associated with each node, down to terminal leaves. 
"""
function leaf_weights!(nw::AnyCuArray)
    cw = CUDA.ones(eltype(nw), 2 * size(nw, 1) + 1, size(nw)[2:3]...)
    blocks = size(nw, 3)
    threads = size(nw, 2)
    @cuda threads = threads blocks = blocks leaf_weights!_kernel!(cw, nw)
    CUDA.synchronize()
    @views lw = cw[size(nw, 1)+1:size(cw, 1), :, 1:size(nw, 3)]
    return (cw, lw)
end

"""
    leaf_weights_functional(nw)

Non-mutating functional implementation for computing cumulative leaf weights.
Uses a purely functional approach with array operations for AD compatibility.
"""
function leaf_weights_functional(nw)
    n_nodes = size(nw, 1)
    n_trees = size(nw, 2) 
    n_batches = size(nw, 3)
    cw_size = 2 * n_nodes + 1
    
    # Initialize cumulative weights array with ones at root (index 1)
    cw = zeros(eltype(nw), cw_size, n_trees, n_batches)
    
    # Set root nodes to 1.0
    cw = _set_roots(cw, n_trees, n_batches)
    
    # Compute cumulative weights using functional operations
    cw = _compute_cumulative_functional(cw, nw, n_nodes)
    
    # Extract leaf weights
    lw = cw[(n_nodes + 1):cw_size, :, :]
    
    return (cw, lw)
end

"""
    _set_roots(cw, n_trees, n_batches)

Helper function to set root nodes to 1.0 in a non-mutating way.
"""
function _set_roots(cw, n_trees, n_batches)
    root_values = ones(eltype(cw), 1, n_trees, n_batches)
    return vcat(root_values, cw[2:end, :, :])
end

"""
    _compute_cumulative_functional(cw, nw, n_nodes)

Functional computation of cumulative weights using array operations.
"""
function _compute_cumulative_functional(cw, nw, n_nodes)
    current_cw = cw
    
    # Process each level of the tree
    for parent_idx in 1:n_nodes
        left_child = 2 * parent_idx
        right_child = 2 * parent_idx + 1
        
        if left_child <= size(current_cw, 1)
            # Compute left child weights
            left_weights = current_cw[parent_idx:parent_idx, :, :] .* nw[parent_idx:parent_idx, :, :]
            current_cw = _update_at_index(current_cw, left_weights, left_child)
        end
        
        if right_child <= size(current_cw, 1)
            # Compute right child weights  
            right_weights = current_cw[parent_idx:parent_idx, :, :] .* (1 .- nw[parent_idx:parent_idx, :, :])
            current_cw = _update_at_index(current_cw, right_weights, right_child)
        end
    end
    
    return current_cw
end

"""
    _update_at_index(arr, new_values, index)

Non-mutating update of array at specific index.
"""
function _update_at_index(arr, new_values, index)
    return vcat(
        arr[1:(index-1), :, :],
        new_values,
        arr[(index+1):end, :, :]
    )
end

