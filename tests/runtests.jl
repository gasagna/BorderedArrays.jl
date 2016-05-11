using Base.Test
using BorderedArrays
import Base.LinAlg: A_ldiv_B!, 
                    At_ldiv_B!, 
                    Ac_ldiv_B!    

# test block vector
let 
    x = BlockVector([1 3 5;
                     2 4 6])
    @test eltype(x) == Int
    for i = 1:length(x)
        x[i] = i+1
        @test x[i] == i+1
        x[i] = i
    end
    @test full(x) == [1, 2, 3, 4, 5, 6]
    @test size(x) == (6, )
    @test_throws BoundsError size(x, 0)
    @test size(x, 1) == 6
    @test size(x, 2) == 1
    @test length(x) == 6
    @test nblocks(x) == 3
    @test blocksize(x) == 2
    @test BorderedArrays._blockstart(x, 1) == 1
    @test BorderedArrays._blockstart(x, 2) == 3
    @test BorderedArrays._blockstart(x, 3) == 5
    @test_throws BoundsError BorderedArrays._blockstart(x, 0)
    @test_throws BoundsError BorderedArrays._blockstart(x, 4)

    # test dot product
    x = BlockVector(2.0*[1 3 5;
                         2 4 6])
    y = BlockVector(0.5*[1 3 5;
                         2 4 6])
    @test dot(x, y) == 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6
end

# test block matrix
let 
    A = BlockDiagonalMatrix([1 3 5;          
                             2 3 4;         
                             2 3 4;         
                             2 4 6])
    @test eltype(A) == Int
    for i = 1:size(A, 1)
        @test A[i, i] == i
    end
    @test full(A) == [1  2  0  0  0  0;
                      2  2  0  0  0  0;
                      0  0  3  3  0  0;
                      0  0  3  4  0  0;
                      0  0  0  0  5  4;
                      0  0  0  0  4  6]                  
    @test size(A) == (6, 6)
    @test_throws BoundsError size(A, 0)
    @test size(A, 1) == 6
    @test size(A, 2) == 6
    @test size(A, 3) == 1
    @test nblocks(A) == 3
    @test blocksize(A) == 2
    @test BorderedArrays._blockstart(A, 1) == 1
    @test BorderedArrays._blockstart(A, 2) == 5
    @test BorderedArrays._blockstart(A, 3) == 9
    @test_throws BoundsError BorderedArrays._blockstart(A, 0)
    @test_throws BoundsError BorderedArrays._blockstart(A, 4)

    # test block setindexing 
    for k = 1:nblocks(A)
        for i = 1:blocksize(A), j = 1:blocksize(A)
            A[k, i, j] = k # these are local indices
        end
    end
    @test full(A) == [1  1  0  0  0  0;
                      1  1  0  0  0  0;
                      0  0  2  2  0  0;
                      0  0  2  2  0  0;
                      0  0  0  0  3  3;
                      0  0  0  0  3  3]
    # test block getindexing                      
    for k = 1:nblocks(A)
        for i = 1:blocksize(A), j = 1:blocksize(A)
            @test A[k, i, j] == k # these are local indices
        end
    end                      
end

# test bordered block vector
let
    x = BorderedVector(BlockVector([1 3 5;
                                    2 4 6]), 7)
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
    for n = 1:10, nb = 1:10
        upp = BlockVector(rand(Float64, n, nb))
        low = 0.0
        x = BorderedVector(upp, low)
        @test size(x) == (n*nb+1, )
        @test size(x, 1) == n*nb+1
        @test size(x, 2) == 1
        @test_throws BoundsError size(x, 0)
        @test full(x) == x
        @test_throws BoundsError x[0]
        @test_throws BoundsError x[n*nb+1 + 1]
        @test eltype(x) == Float64
    end
end

# test bordered block matrix
let 
    A = BlockDiagonalMatrix([1 2 3;          
                     2 3 4;         
                     2 3 4;         
                     1 2 3])
    b = BlockVector([1 3 5;
                     2 4 6])
    c = BlockVector([6 4 2;
                     5 3 1])
    d = 0
    M = BorderedMatrix(A, b, c, d)
    @test size(M) == (7, 7)
    @test size(M, 1) == size(M, 2) == 7
    @test full(M) == [1  2  0  0  0  0  1;
                      2  1  0  0  0  0  2;
                      0  0  2  3  0  0  3;
                      0  0  3  2  0  0  4;
                      0  0  0  0  3  4  5;
                      0  0  0  0  4  3  6;
                      6  5  4  3  2  1  0] == M
    @test_throws BoundsError M[0, 0]
    @test_throws BoundsError M[1, 8]
    @test_throws BoundsError M[8, 1]
    @test_throws BoundsError M[8, 8]
    @test eltype(M) == Int

    # more sizes
    for n = 1:10, nb = 1:10
        A = BlockDiagonalMatrix(rand(Float64, n*n, nb))
        b = BlockVector(rand(Float64, n, nb))
        c = BlockVector(rand(Float64, n, nb))
        d = 0.0
        M = BorderedMatrix(A, b, c, d)
        @test size(M) == (n*nb+1, n*nb+1)
        @test size(M, 1) == size(M, 2) == n*nb+1
        @test full(M) == M
        @test_throws BoundsError M[0, 0]
        @test_throws BoundsError M[1, n*nb+2]
        @test_throws BoundsError M[n*nb+2, 1]
        @test_throws BoundsError M[n*nb+2, n*nb+2]
        @test eltype(M) == Float64
    end
end  
       
# test solution of block diagonal matrix 
let 
    # demo block matrix
    srand(0)
    for typ in (Float64, Complex128)
        for (ch, fname) in zip(('N', 'T'), (A_ldiv_B!, At_ldiv_B!))   
            for n = 1:10, nb = 1:10
                A = BlockDiagonalMatrix(rand(typ, n*n, nb))
                b = BlockVector(rand(typ, n, nb))
            
                # take copy for testing
                Ad = full(A)
                bd = full(b)

                # solutions
                xd, td, _, _, _ = @timed fname(lufact!(Ad), bd)
                x, t, _, _, _   = @timed fname(lufact!(A), b)

                # speed up
                # speedup = td/t
                # @printf "%s %3d - %3d - %5.2f\n" typ n nb speedup
                
                # check
                @test xd ≈ x
            end
        end
    end
end

# test solution of bordered matrix
let 
    srand(0)
    for typ in [Float64, Complex128]
        for n = 1:10, nb = 1:10
            # demo matrices
            A = BlockDiagonalMatrix(rand(typ, n*n, nb))
            b = BlockVector(rand(typ, n, nb))
            c = BlockVector(rand(typ, n, nb))
            d = rand(typ)
            M = BorderedMatrix(A, b, c, d)
            Md = full(M)

            # demo array
            x = BlockVector(rand(typ, n, nb))
            y = rand(typ)
            r = BorderedVector(x, y)
            rd = full(r)

            # solutions
            # r = A_ldiv_B!(M, r, :BED)
            # rd = Md\rd

            rd, td, _, _, _ = @timed Md\rd
            r, t, _, _, _   = @timed A_ldiv_B!(M, r, :BED)

            # # speed up
            # speedup = td/t
            # @printf "%s %3d - %3d - %5.2f\n" typ n nb speedup

            # check
            @test r ≈ rd
        end
    end
end