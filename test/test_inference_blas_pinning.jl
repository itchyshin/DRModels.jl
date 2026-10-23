using DRModels, Test
using LinearAlgebra: BLAS

@testset "inference BLAS pinning is composable across callers" begin
    original_blas = BLAS.get_num_threads()
    a = nothing
    b = nothing
    release_a = Channel{Nothing}(1)
    release_b = Channel{Nothing}(1)

    try
        # This test does not need multiple scheduler threads: the channels yield
        # on one thread and also coordinate two actual tasks on larger runtimes.
        BLAS.set_num_threads(2)
        @test BLAS.get_num_threads() == 2

        a_entered = Channel{Nothing}(1)
        b_entered = Channel{Nothing}(1)
        a = Threads.@spawn DRModels._with_pinned_blas(true) do
            put!(a_entered, nothing)
            take!(release_a)
        end
        take!(a_entered)
        @test BLAS.get_num_threads() == 1

        b = Threads.@spawn DRModels._with_pinned_blas(true) do
            put!(b_entered, nothing)
            take!(release_b)
        end
        take!(b_entered)
        @test BLAS.get_num_threads() == 1

        # A has finished, but B is still in its concurrent region. The pin must
        # remain active until the final concurrent caller exits.
        put!(release_a, nothing)
        fetch(a)
        @test BLAS.get_num_threads() == 1

        put!(release_b, nothing)
        fetch(b)
        @test BLAS.get_num_threads() == 2

        DRModels._with_pinned_blas(true) do
            @test BLAS.get_num_threads() == 1
            DRModels._with_pinned_blas(true) do
                @test BLAS.get_num_threads() == 1
            end
            @test BLAS.get_num_threads() == 1
        end
        @test BLAS.get_num_threads() == 2

        DRModels._with_pinned_blas(false) do
            @test BLAS.get_num_threads() == 2
        end
        @test BLAS.get_num_threads() == 2

        # A scope that starts with BLAS already pinned participates in the same
        # nested lifetime accounting and restores that original one-thread state.
        BLAS.set_num_threads(1)
        DRModels._with_pinned_blas(true) do
            @test BLAS.get_num_threads() == 1
            DRModels._with_pinned_blas(true) do
                @test BLAS.get_num_threads() == 1
            end
        end
        @test BLAS.get_num_threads() == 1
        BLAS.set_num_threads(2)

        @test_throws ErrorException DRModels._with_pinned_blas(true) do
            @test BLAS.get_num_threads() == 1
            error("intentional BLAS-pinning exception")
        end
        @test BLAS.get_num_threads() == 2
    finally
        # Never strand a task if an assertion or unexpected test setup failure
        # interrupts the coordinating path.
        isready(release_a) || put!(release_a, nothing)
        isready(release_b) || put!(release_b, nothing)
        a === nothing || try fetch(a) catch; end
        b === nothing || try fetch(b) catch; end
        BLAS.set_num_threads(original_blas)
    end
end
