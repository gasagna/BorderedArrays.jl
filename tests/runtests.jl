using Base.Test
using BorderedArrays
import Base.LinAlg: A_ldiv_B!,
                    At_ldiv_B!,
                    Ac_ldiv_B!

# test bordered block vector
let
    x = BorderedVector([1, 2, 3, 4, 5, 6], 7)
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
end

# test bordered matrix
let
    A = [1 2 3 1;
         2 3 4 1;
         2 3 4 1;
         1 2 3 1]
    b = [1, 2, 3, 4]
    c = [4, 3, 2, 1]
    d = 0
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
        rv = A_ldiv_B!(Mv, full(r), :BED)
        r  = A_ldiv_B!(M,  r,       :BED)
        rd = Md\rd

        # check
        @test r ≈ rd
        @test r ≈ rv
    end
end