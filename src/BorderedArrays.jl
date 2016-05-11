# ------------------------------------------------------------------- #
# Copyright 2015-2016, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
module BorderedArrays

using Devectorize

import Base: convert,
             lufact!,
             full,
             size,
             getindex,
             setindex!,
             length,
             linearindexing,
             LinearSlow, 
             dot

import Base.LinAlg: A_ldiv_B!,
                    At_ldiv_B!,
                    Ac_ldiv_B!

export BorderedMatrix,
       BorderedVector,
       BlockDiagonalMatrix,
       BlockVector,
       block,
       nblocks,
       blocksize,
       upper, 
       lower

# ~~~ Block Vector ~~~

# type to store a vector-of-vectors
type BlockVector{T<:Number} <: AbstractVector{T}
    data::Matrix{T} # the i-th column is i-th vector
end

# block properties
function _blockstart(v::BlockVector, i::Integer) 
    0 < i <= nblocks(v) || throw(BoundsError("wrong input index"))
    (i-1)*blocksize(v) + 1
end
nblocks(v::BlockVector) = size(v.data, 2)
blocksize(v::BlockVector) = size(v.data, 1)

# array interface
size(v::BlockVector) = (length(v), )
length(v::BlockVector) = length(v.data)
eltype{T}(v::BlockVector{T}) = T
linearindexing(v::BlockVector) = LinearSlow()
getindex(v::BlockVector, i::Integer) = v.data[i]
setindex!(v::BlockVector, val, i::Integer) = (v.data[i] = val)

# dot product
function dot{T}(v::BlockVector{T}, u::BlockVector{T})
    length(v) == length(u) || 
        error(DimensionMismatch("block vectors must have same length"))
    vdata, udata = v.data, u.data
    out = zero(T)
    @simd for i = 1:length(v)
        @inbounds out += vdata[i]*udata[i]
    end
    out
end


# ~~~ Block Diagonal Matrix ~~~

# type to store a block diagonal matrix with square blocks
type BlockDiagonalMatrix{T<:Number} <: AbstractMatrix{T}
    data::Matrix{T} # column i is the i-th block stored in a column-major format
    function BlockDiagonalMatrix(data::Matrix{T})
        # can rearrange data in a diagonal block matrix with square blocks?
        n = sqrt(size(data, 1))
        ceil(n) - floor(n) == 0  || error("error in input data")
        new(data)
    end
end
BlockDiagonalMatrix{T}(data::Matrix{T}) = BlockDiagonalMatrix{T}(data)

# pointer to first element of the ith block
function _blockstart(v::BlockDiagonalMatrix, i::Integer) 
    0 < i <= nblocks(v) || throw(BoundsError("wrong input index"))
    (i-1)*blocksize(v)^2 + 1
end

# Number of blocks and block size
nblocks(v::BlockDiagonalMatrix) = size(v.data, 2)
blocksize(v::BlockDiagonalMatrix) = Int(sqrt(size(v.data, 1)))

# array interface
size(v::BlockDiagonalMatrix) = (g = nblocks(v)*blocksize(v); (g, g))
eltype{T}(v::BlockDiagonalMatrix{T}) = T
linearindexing(v::BlockDiagonalMatrix) = LinearSlow()

# Global indexing 
function getindex{T}(v::BlockDiagonalMatrix{T}, i::Integer, j::Integer)
    n = blocksize(v)
    # block and local coordinates
    iglob, iloc = divrem(i+n-1, n)
    jglob, jloc = divrem(j+n-1, n)
    # if outside of diagonal return 0
    val = iglob != jglob ? zero(T) : v.data[jloc*n + iloc+1, jglob]
    return val
end

# Block indexing
setindex!{T}(v::BlockDiagonalMatrix{T}, val, k, i::Integer, j::Integer) = 
    v.data[i + (j-1)*blocksize(v),  Int(k)] = val
getindex{T}(v::BlockDiagonalMatrix{T}, k, i::Integer, j::Integer) =
    v.data[i + (j-1)*blocksize(v),  Int(k)]


# ~~~ Bordered Vector ~~~

# Type to store a vector bordered by a final value
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
eltype{T}(v::BlockDiagonalMatrix{T}) = T
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


# ~~~ Utilities ~~~

# collect to a DenseArray - useful for debugging solvers
full(v::Union{BlockVector, BorderedVector}) = collect(v)
function full{T}(M::Union{BlockDiagonalMatrix{T}, BorderedMatrix{T}})
    m, n = size(M)
    Mdense = zeros(T, m, n)
    for i = 1:m, j = 1:n
        Mdense[i, j] = M[i, j]
    end
    Mdense
end


# ~~~ Solvers ~~~

const liblapack = Base.liblapack_name
import Base.LinAlg: BlasInt
import Base.LAPACK: getrs!, getrf!

# ~~~ Linear algebra for block systems ~~~
# copied form base
immutable BlockLU{T} <: Factorization{T}
    factors::BlockDiagonalMatrix{T}
    ipiv::BlockVector{BlasInt}
    info::Vector{BlasInt}
end    

# in-place LU factorisation with pivoting of a square block diagonal matrix 
function lufact!{T}(A::BlockDiagonalMatrix{T})
    A, ipiv, info = getrf!(A)
    return BlockLU{T}(A, ipiv, info)
end

# A_ldiv_B business for block lu factorisation
for typ in (Float64, Complex128)
    for (ch, fname) in zip(('N', 'T'), (:A_ldiv_B!, :At_ldiv_B!))
        @eval function $fname{typ}(F::BlockLU{typ}, B::BlockVector{typ})
            # check there is no singular block
            m, k = findmax(F.info)
            m > 0 && throw(SingularException("block $k is singular"))
            getrs!($ch, F.factors, F.ipiv, F.info, B) 
        end
    end
end

#  Custom LAPACK wrappers for block matrices 
for (getrs, getrf, elty) in ((:zgetrs_, :zgetrf_, Complex128), 
                             (:dgetrs_, :dgetrf_, Float64))
    #DGETRF computes an LU factorization of a general M-by-N matrix A
    # using partial pivoting with row interchanges.
    #
    # The factorization has the form
    #    A = P * L * U
    # where P is a permutation matrix, L is lower triangular with unit
    # diagonal elements (lower trapezoidal if m > n), and U is upper
    # triangular (upper trapezoidal if m < n).
    #
    # This is the right-looking Level 3 BLAS version of the algorithm.
    @eval function getrf!(A::BlockDiagonalMatrix{$elty})
        n, nb = blocksize(A), nblocks(A)
        lda  = n
        info = Vector{BlasInt}(nb)
        ipiv = BlockVector(Array(BlasInt, n, nb))
        for k = 1:nb
            ccall(($(string(getrf)), liblapack), Void,
                  (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  &n, &n, pointer(A.data, _blockstart(A, k)), &lda, 
                  pointer(ipiv.data, _blockstart(ipiv, k)), pointer(info, k))
            info[k] < 0 && throw(ArgumentError("invalid argument #$(info[k])"))
            info[k] > 0 && throw(SingularException("block $k is singular"))
        end
        A, ipiv, info
    end

    # DGETRS solves a system of linear equations
    # A * X = B  or  A**T * X = B
    # with a general N-by-N matrix A using the LU factorization 
    # computed by DGETRF
    @eval function getrs!(trans::Char, 
                          A::BlockDiagonalMatrix{$elty}, 
                          ipiv::BlockVector{BlasInt}, 
                          info::Vector{BlasInt},
                          B::BlockVector{$elty})
        Base.LAPACK.chktrans(trans)
        n = blocksize(A)
        n != blocksize(B) && throw(DimensionMismatch("block sizes must match"))
        for k = 1:nblocks(A)
            ccall(($(string(getrs)), liblapack), Void,
                  (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                  &trans, &n, &1, pointer(A.data, _blockstart(A, k)), 
                  &n, pointer(ipiv.data, _blockstart(ipiv, k)), 
                  pointer(B.data, _blockstart(B, k)), &n, pointer(info, k))
            info[k] < 0 && throw(ArgumentError("invalid argument #$(info[k])"))
        end
        B
    end
end


# ~~~ Linear algebra for bordered systems ~~~

# Generic entry point for bordered systems
function A_ldiv_B!(M::BorderedMatrix, r::BorderedVector, alg::Symbol=:BED)
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

    throw(ArgumentError("invalid `alg` parameter"))
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
    lu = lufact!(A)

    # step 1 solve A' * w = c - overwrite c with w
    w = At_ldiv_B!(lu, c)

    # step 2: compute δ⁺ = d - w'*b
    δ⁺ = d - dot(w, b)

    # step 3: compute y = (g - w'*f)/δ⁺
    r._₂ = (g - dot(w, f))/δ⁺    

    # step 4: solve A*x = f - b*y
    @simd for i in 1:length(f)
        @inbounds f[i] -= b[i]*r._₂
    end
    A_ldiv_B!(lu, f)

    # return solution
    r
end

end