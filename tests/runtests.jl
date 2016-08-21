using BenchmarkTools
using Base.Test
using BorderedMatrices
import Base.LinAlg: A_ldiv_B!,
                    At_ldiv_B!,
                    Ac_ldiv_B!

# test bordered block vector
let
    x = BorderedVector([1, 2, 3, 4, 5, 6], 7.0)
    @test length(x) == 7
    @test size(x) == (7, )
    @test_throws BoundsError size(x, 0)
    @test size(x, 1) == 7
    @test size(x, 2) == 1
    for i = 1:length(x)
        @test x[i] == i
    end
    @test full(x) == x == [1, 2, 3, 4, 5, 6, 7]
    @test_throws BoundsError x[0]
    @test_throws BoundsError x[8]
    @test eltype(x) == Int

    y = (x[1] = 2)
    @test y == 2
    @test x[1] == 2
    @test_throws BoundsError x[0] = 1
    @test_throws BoundsError x[8] = 1

    # more sizes
    for n = 5:100
        upp = rand(Float64, n)
        low = rand()
        x = BorderedVector(upp, low)
        @test size(x) == (n+1, )
        @test size(x, 1) == n+1
        @test size(x, 2) == 1
        @test_throws BoundsError size(x, 0)
        @test full(x) == x
        @test_throws BoundsError x[0]
        @test_throws BoundsError x[n+1 + 1]
        @test eltype(x) == Float64
    end

    # test fancy setindexing
    x = BorderedVector([1, 2, 3, 4], 5)
    x[1:end] = 0
    @test x == [0, 0, 0, 0, 0]

    x = BorderedVector([1, 2, 3, 4], 5)
    x[[1, 5]] = 0
    @test x == [0, 2, 3, 4, 0]

    x = BorderedVector([1, 2, 3, 4], 5)
    x[:] = 1
    @test x == [1, 1, 1, 1, 1]

    # test fancy getindexing
    x = BorderedVector([1, 2, 3, 4], 5)
    @test x[:] == [1, 2, 3, 4, 5]
    @test x[1:3] == [1, 2, 3]
    @test x[4:end] == [4, 5]
end

# test bordered matrix
let
    A = [1 2 3 1;
         2 3 4 1;
         2 3 4 1;
         1 2 3 1]
    b = [1, 2, 3, 4]
    c = [4, 3, 2, 1]
    d = 0.0 # will convert to int
    M = BorderedMatrix(A, b, c, d)
    @test size(M) == (5, 5)
    @test size(M, 1) == size(M, 2) == 5
    @test full(M) == [1  2  3  1  1;
                      2  3  4  1  2;
                      2  3  4  1  3;
                      1  2  3  1  4;
                      4  3  2  1  0] == M
    @test_throws BoundsError M[0, 0]
    @test_throws BoundsError M[1, 6]
    @test_throws BoundsError M[6, 1]
    @test_throws BoundsError M[6, 6]
    @test eltype(M) == Int

    # test setindex
    for i = 1:5, j = 1:5
        M[i, j] = i*j
    end
    @test M == [i*j for i = 1:5, j = 1:5]
end

# test copy/similar for vector
let
    x = BorderedVector([1, 2, 3, 4, 5, 6], 7)
    y = copy(x)
    @test y == [1, 2, 3, 4, 5, 6, 7]
    y[1] = 5
    @test x[1] == 1

    s = similar(x)
    @test isa(s, BorderedVector)
    for fun in [length, size, eltype]
        @test fun(s) == fun(x)
    end
end

# test copy/similar for matrix
let
    A = [1 2 3 1;
         2 3 4 1;
         2 3 4 1;
         1 2 3 1]
    b = [1, 2, 3, 4]
    c = [4, 3, 2, 1]
    d = 0
    M = BorderedMatrix(A, b, c, d)
    N = copy(M)
    @test full(N) == [1  2  3  1  1;
                      2  3  4  1  2;
                      2  3  4  1  3;
                      1  2  3  1  4;
                      4  3  2  1  0] == M
    N._₁₁[1, 1] = 5
    @test M[1, 1] == 1

    N = similar(M)
    @test isa(N, BorderedMatrix)
    for fun in [size, eltype]
        @test fun(N) == fun(M)
    end
end

# test solution of bordered matrix
let
    srand(0)
    for n = 10:100
        # demo matrices
        A = rand(n, n)
        b = rand(n)
        c = rand(n)
        d = rand()
        M  = BorderedMatrix(A, b, c, d)
        Mv = BorderedMatrix(copy(A), copy(b), copy(c), copy(d))
        Md = full(M)

        # demo array
        x = rand(n)
        y = rand()
        r = BorderedVector(x, y)
        rd = full(r)

        # solutions
        rBEM  = A_ldiv_B!(copy(M), copy(r), :BEM)
        rBED  = A_ldiv_B!(copy(M), copy(r), :BED)
        rd = Md\rd

        # check
        @test rBED ≈ rd
        @test rBEM ≈ rd
    end
end

# test BEM solves correctly system with ill-conditioned A
let
    A = Float64[1 0     0;
                0 1     0;
                0 0 1e-32]
    b = Float64[1, 1, 1]
    c = Float64[2, 2, 2]
    d = 1.0
    M = BorderedMatrix(A, b, c, d)
    r = BorderedVector([3.0, 4.0, 5.0], 1.0)

    @test cond(full(M)) < 11
    @test cond(full(A)) > 1e31

    xBEM = A_ldiv_B!(copy(M), copy(r), :BEM)
    xBED = A_ldiv_B!(copy(M), copy(r), :BED)
    x = full(M)\full(r)

    @test x == xBEM
    @test x != xBED
end