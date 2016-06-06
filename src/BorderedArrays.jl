# ------------------------------------------------------------------- #
# Copyright 2015-2016, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
__precompile__()
module BorderedArrays

import Base: convert,
             lufact!,
             full,
             size,
             getindex,
             setindex!,
             length,
             linearindexing,
             LinearSlow, 
             dot, 
             copy,
             similar
             
import Base.LinAlg: A_ldiv_B!,
                    At_ldiv_B!,
                    Ac_ldiv_B!

export BorderedMatrix,
       BorderedVector

# ~~~ Bordered Vector ~~~

# Type to store a vector(s) bordered by a final value(s)
type BorderedVector{T<:Number, V<:AbstractVector} <: AbstractVector{T}
    _₁::V # main part
    _₂::T # last element
    BorderedVector(v₁::AbstractVector{T}, v₂::T) = new(v₁, v₂)
end
BorderedVector{T}(v₁::AbstractVector{T}, v₂::T) = 
    BorderedVector{T, typeof(v₁)}(v₁, v₂) 

# array interface
length(v::BorderedVector) = length(v._₁) + 1
size(v::BorderedVector) = (length(v), )
eltype{T}(v::BorderedVector{T}) = T
linearindexing(v::BorderedVector) = LinearSlow()

function getindex(v::BorderedVector, i::Integer) 
    1 <= i <= length(v._₁) && return v._₁[i]
    i == length(v._₁) + 1  && return v._₂
    throw(BoundsError())
end
function setindex!(v::BorderedVector, val, i::Integer) 
    1 <= i <= length(v._₁) && (v._₁[i] = val; return v)
    i == length(v._₁) + 1  && (v._₂ = val; return v)
    throw(BoundsError())
end

# copy and similar
copy(v::BorderedVector) = BorderedVector(copy(v._₁), v._₂)
similar(v::BorderedVector) = BorderedVector(similar(v._₁), zero(v._₂))

# collect to a DenseArray - useful for debugging solvers
full(v::BorderedVector) = collect(v)

# ~~~ Bordered Matrix ~~~

# type to store a square matrix bordered by two vectors and a scalar
type BorderedMatrix{T<:Number, 
                    M<:AbstractMatrix, V<:AbstractVector} <: AbstractMatrix{T}
    _₁₁::M # main top left - any matrix
    _₁₂::V # vertical right vector - any vector
    _₂₁::V # horizontal bottom vector - any vector
    _₂₂::T # bottom right element - a scalar
    function BorderedMatrix(M₁₁::AbstractMatrix{T}, 
                            M₁₂::AbstractVector{T}, 
                            M₂₁::AbstractVector{T},
                            M₂₂::T)
        size(M₁₁) == (length(M₁₂), length(M₂₁)) || 
            throw(DimensionMismatch("inconsistent input size"))
        new(M₁₁, M₁₂, M₂₁, M₂₂)
    end
end
BorderedMatrix{T}(M₁₁::AbstractMatrix{T}, 
                  M₁₂::AbstractVector{T}, 
                  M₂₁::AbstractVector{T},
                  M₂₂::T) = 
    BorderedMatrix{T, typeof(M₁₁), typeof(M₁₂)}(M₁₁, M₁₂, M₂₁, M₂₂)

# array interface
eltype{T}(v::BorderedMatrix{T}) = T
size(M::BorderedMatrix) = (size(M._₁₁, 1) + 1, size(M._₁₁, 2) + 1)
linearindexing(v::BorderedMatrix) = LinearSlow()

function getindex(M::BorderedMatrix, i::Integer, j::Integer)
    m, n = size(M)
    if i < m
        if j < n
            return M._₁₁[i, j]
        elseif j == n
            return M._₁₂[i]
        end
    elseif i == m
        if j < n
            return M._₂₁[j]
        elseif j == n
            return M._₂₂
        end
    end
    throw(BoundsError())
end            

# copy/similar
copy(M::BorderedMatrix) = 
    BorderedMatrix(copy(M._₁₁), copy(M._₁₂), copy(M._₂₁), M._₂₂)

similar(M::BorderedMatrix) = 
    BorderedMatrix(similar(M._₁₁), similar(M._₁₂), similar(M._₂₁), zero(M._₂₂))

# collect to a DenseArray - useful for debugging solvers
function full{T}(M::BorderedMatrix{T})
    m, n = size(M)
    Mdense = zeros(T, m, n)
    for i = 1:m, j = 1:n
        Mdense[i, j] = M[i, j]
    end
    Mdense
end


# ~~~ Linear algebra for bordered systems ~~~
function A_ldiv_B!(M::BorderedMatrix, r::BorderedVector, alg::Symbol=:BEM)
    # Solve the system
    #         
    #     M * z = r
    # 
    # where
    #         /  A   b \
    #     M = |         |
    #         \  cᵀ  d /
    # and 
    #         / f \
    #     r = |   |
    #         \ g /
    #
    # by overwriting the solution in r

    # checks
    size(M, 1) == size(M, 2) ||
        throws(DimensionMismatch("matrix must be square"))
    size(M, 1) == length(r) || 
        throws(DimensionMismatch("inner dimensions must agree"))

    # Select factorisation algorithm
    alg == :BED && return alg_BED!(M, r)
    alg == :BEM && return alg_BEM!(M, r)

    throw(ArgumentError("invalid `alg` parameter"))
end

# solve bordered system with block elimination method
function alg_BEM!(M::BorderedMatrix, r::BorderedVector)
    # rename variables
    A = M._₁₁ # AbstractMatrix
    b = M._₁₂ # AbstractVector
    c = M._₂₁ # AbstractVector
    d = M._₂₂ # Scalar
    f = r._₁  # AbstractVector
    g = r._₂  # Scalar

    # step 0: factorise A
    Aᶠ = lufact!(A)

    # step 1: solve Aᵀw = c
    w = At_ldiv_B!(Aᶠ, copy(c))

    # step 2: compute δ⁺ = d - w'*b
    δ⁺ = d - dot(w, b)

    # step 3: solve Av = b
    v = At_ldiv_B!(Aᶠ, copy(b))

    # step 4: 
    δ = d - dot(c, v)

    # step 5:
    y₁ = (g - dot(w, f))/δ⁺

    # step 6: f₁ = f - b*y₁ - aliased to f
    @simd for i in 1:length(f)
        @inbounds f[i] -= b[i]*y₁
    end

    # step 7: 
    g₁ = g - d*y₁

    # step 8
    ξ = A_ldiv_B!(Aᶠ, f)

    # step 9
    y₂ = (g₁ - dot(c, ξ))/δ

    # step 10: x = ξ - v*y₂ - aliased to r._₁
    x = r._₁
    @simd for i in 1:length(x)
        @inbounds x[i] = ξ[i] - v[i]*y₂
    end

    # step 11 - y = y₁ + y₂
    r._₂ = y₁ + y₂

    r
end

# solve bordered system using doolittle factorisation
function alg_BED!(M::BorderedMatrix, r::BorderedVector)
    # rename variables
    A = M._₁₁ # AbstractMatrix
    b = M._₁₂ # AbstractVector
    c = M._₂₁ # AbstractVector
    d = M._₂₂ # Scalar
    f = r._₁  # AbstractVector
    g = r._₂  # Scalar

    # step 0: factorise A
    Aᶠ = lufact!(A)

    # step 1 solve A' * w = c - overwrite c with w
    w = At_ldiv_B!(Aᶠ, c)

    # step 2: compute δ⁺ = d - w'*b
    δ⁺ = d - dot(w, b)

    # step 3: compute y = (g - w'*f)/δ⁺
    r._₂ = (g - dot(w, f))/δ⁺    

    # step 4: solve A*x = f - b*y
    @simd for i in 1:length(f)
        @inbounds f[i] -= b[i]*r._₂
    end
    A_ldiv_B!(Aᶠ, f)

    # return solution
    r
end

end