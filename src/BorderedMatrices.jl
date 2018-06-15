# ------------------------------------------------------------------- #
# Copyright 2015-2016, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
__precompile__()
module BorderedMatrices

export BorderedMatrix, BorderedVector

# ~~~ Bordered Vector ~~~

# Type to store a vector(s) bordered by a final value(s)
mutable struct BorderedVector{T<:Number, V<:AbstractVector{T}} <: AbstractVector{T}
    _1::V # main part
    _2::T # last element
    BorderedVector(v₁::V, v₂::Real) where {T, V<:AbstractVector{T}} = 
    	new{T, V}(v₁, convert(T, v₂))
end

# array interface
Base.size(v::BorderedVector) = (length(v._1) + 1, )
Base.IndexStyle(v::BorderedVector) = Base.IndexLinear()

function Base.getindex(v::BorderedVector, i::Integer) 
    1 <= i <= length(v._1) && return v._1[i]
    i == length(v._1) + 1  && return v._2
    throw(BoundsError())
end
function Base.setindex!(v::BorderedVector, val, i::Integer) 
    1 <= i <= length(v._1) && (v._1[i] = val; return v)
    i == length(v._1) + 1  && (v._2 = val; return v)
    throw(BoundsError())
end

# copy and similar
Base.copy(v::BorderedVector) = BorderedVector(copy(v._1), v._2)
Base.similar(v::BorderedVector) = BorderedVector(similar(v._1), zero(v._2))

# collect to a DenseArray - useful for debugging solvers
Base.full(v::BorderedVector) = collect(v)

# ~~~ Bordered Matrix ~~~

# type to store a square matrix bordered by two vectors and a scalar
mutable struct BorderedMatrix{T<:Number, 
                              M<:AbstractMatrix{T}, 
                              V<:AbstractVector{T}} <: AbstractMatrix{T}
    _11::M # main top left - any matrix
    _12::V # vertical right vector - any vector
    _21::V # horizontal bottom vector - any vector
    _22::T # bottom right element - a scalar
    function BorderedMatrix(M₁₁::M,
                            M₁₂::V,
                            M₂₁::V,
                            M₂₂::Real) where {T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
        size(M₁₁) == (length(M₁₂), length(M₂₁)) || 
            throw(DimensionMismatch("inconsistent input size"))
        new{T, M, V}(M₁₁, M₁₂, M₂₁, convert(T, M₂₂))
    end
end

# array interface
Base.size(M::BorderedMatrix) = (size(M._11, 1) + 1, size(M._11, 2) + 1)
Base.IndexStyle(v::BorderedMatrix) = Base.IndexCartesian()

function Base.getindex(M::BorderedMatrix, i::Integer, j::Integer)
    m, n = size(M)
    if i < m
        if j < n
            return M._11[i, j]
        elseif j == n
            return M._12[i]
        end
    elseif i == m
        if j < n
            return M._21[j]
        elseif j == n
            return M._22
        end
    end
    throw(BoundsError())
end        

function Base.setindex!(M::BorderedMatrix, val, i::Integer, j::Integer)
    m, n = size(M)
    if i < m
        if j < n
            return M._11[i, j] = val
        elseif j == n
            return M._12[i] = val
        end
    elseif i == m
        if j < n
            return M._21[j] = val
        elseif j == n
            return M._22 = val
        end
    end
    throw(BoundsError())
end         

# copy/similar
Base.copy(M::BorderedMatrix) = 
    BorderedMatrix(copy(M._11), copy(M._12), copy(M._21), M._22)

Base.similar(M::BorderedMatrix) = 
    BorderedMatrix(similar(M._11), similar(M._12), similar(M._21), zero(M._22))

# collect to a DenseArray - useful for debugging solvers
function Base.full{T}(M::BorderedMatrix{T})
    m, n = size(M)
    Mdense = zeros(T, m, n)
    for i = 1:m, j = 1:n
        Mdense[i, j] = M[i, j]
    end
    Mdense
end


# ~~~ Linear algebra for bordered systems ~~~
struct BorderedMatrixLU{T<:Number, 
                        M<:Factorization{T}, 
                        V<:AbstractVector{T}} <: Factorization{T}
    _11::M # the parent matrix factorisation
    _12::V # the right bordering vector
    _21::V # the bottom bordering vector
    _22::T # the bordering scalar
end

# in-place and out-of-place factorisations
Base.lufact!(M::BorderedMatrix) = 
    BorderedMatrixLU(lufact!(M._11), M._12, M._21, M._22)

Base.lufact(M::BorderedMatrix) = 
    BorderedMatrixLU(lufact(M._11), M._12, M._21, M._22)

function Base.LinAlg.A_ldiv_B!(M::BorderedMatrix, r::BorderedVector, alg::Symbol=:BEM, inplace::Bool=true)
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
    size(M, 1) == size(M, 2) ||
        throws(DimensionMismatch("matrix must be square"))
    size(M, 1) == length(r) || 
        throws(DimensionMismatch("inner dimensions must agree"))

    return inplace ? A_ldiv_B!(lufact!(M), r, alg) : A_ldiv_B!(lufact(M), r, alg)
end

function Base.LinAlg.A_ldiv_B!(MLU::BorderedMatrixLU, r::BorderedVector, alg::Symbol=:BEM)
    # Select factorisation algorithm
    alg == :BED && return alg_BED!(MLU, r)
    alg == :BEM && return alg_BEM!(MLU, r)
    throw(ArgumentError("invalid `alg` parameter"))
end

# solve bordered system with block elimination method
function alg_BEM!(MLU::BorderedMatrixLU, r::BorderedVector)
    # rename variables
    Aᶠ = MLU._11 # factorisation of AbstractMatrix
    b  = MLU._12 # AbstractVector
    c  = MLU._21 # AbstractVector
    d  = MLU._22 # Scalar
    f  = r._1    # AbstractVector
    g  = r._2    # Scalar

    # step 1: solve Aᵀw = c
    w = At_ldiv_B!(Aᶠ, copy(c))        # allocation

    # step 2: compute δ⁺ = d - w'*b
    δ⁺ = d - dot(w, b)

    # step 3: solve Av = b
    v = A_ldiv_B!(Aᶠ, copy(b))         # allocation

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

    # step 10: x = ξ - v*y₂ - aliased to r._1
    x = r._1
    @simd for i in 1:length(x)
        @inbounds x[i] = ξ[i] - v[i]*y₂
    end

    # step 11 - y = y₁ + y₂
    r._2 = y₁ + y₂

    r
end

# solve bordered system using doolittle factorisation
function alg_BED!(MLU::BorderedMatrixLU, r::BorderedVector)
    # rename variables
    Aᶠ = MLU._11 # Factorization of AbstractMatrix
    b  = MLU._12 # AbstractVector
    c  = MLU._21 # AbstractVector
    d  = MLU._22 # Scalar
    f  = r._1  # AbstractVector
    g  = r._2  # Scalar

    # step 1 solve A' * w = c - overwrite c with w
    w = At_ldiv_B!(Aᶠ, c)

    # step 2: compute δ⁺ = d - w'*b
    δ⁺ = d - dot(w, b)

    # step 3: compute y = (g - w'*f)/δ⁺
    r._2 = (g - dot(w, f))/δ⁺    

    # step 4: solve A*x = f - b*y
    @simd for i in 1:length(f)
        @inbounds f[i] -= b[i]*r._2
    end
    A_ldiv_B!(Aᶠ, f)

    # return solution
    r
end

end