using DRModels, Test

const BinomialLaplace = DRModels

function stable_binomial_reference(s, n, logchoose, eta)
    x = BigFloat(eta)
    log1pexp = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
    p = if x >= 0
        1 - exp(-x) / (1 + exp(-x))
    else
        exp(x) / (1 + exp(x))
    end
    value = BigFloat(n) * log1pexp - BigFloat(s) * x - BigFloat(logchoose)
    d1 = BigFloat(n) * p - BigFloat(s)
    d2 = BigFloat(n) * p * (1 - p)
    d3 = d2 * (1 - 2p)
    return value, d1, d2, d3
end

@testset "Binomial Laplace kernels match the stable logit likelihood" begin
    setprecision(BigFloat, 256) do
        cases = ((0, 1), (1, 2), (2, 2))
        eta_values = (-700.0, -31.0, -30.25, -29.0, 0.4, 29.0,
                      30.25, 31.0, 700.0)

        for (s, n) in cases
            logchoose = log(binomial(n, s))
            aux = (s = [s], ntr = [n], logchoose = [logchoose])
            for eta in eta_values
                expected = stable_binomial_reference(s, n, logchoose, eta)
                actual = BinomialLaplace._laplace_v123(Val(:binomial), aux, 1, eta)

                @test isapprox(actual[1], Float64(expected[1]); atol = 1e-12, rtol = 1e-12)
                @test isapprox(actual[2], Float64(expected[2]); atol = 1e-14, rtol = 1e-12)
                @test isapprox(actual[3], Float64(expected[3]); atol = 1e-14, rtol = 1e-12)
                @test isapprox(actual[4], Float64(expected[4]); atol = 1e-14, rtol = 1e-12)

                @test BinomialLaplace._laplace_value(Val(:binomial), aux, 1, eta) ≈ actual[1]
                @test BinomialLaplace._laplace_d1(Val(:binomial), aux, 1, eta) ≈ actual[2]
                @test BinomialLaplace._laplace_d2(Val(:binomial), aux, 1, eta) ≈ actual[3]
                @test BinomialLaplace._laplace_d3(Val(:binomial), aux, 1, eta) ≈ actual[4]
                d12 = BinomialLaplace._laplace_d12(Val(:binomial), aux, 1, eta)
                @test d12[1] ≈ actual[2]
                @test d12[2] ≈ actual[3]
            end
        end

        # Independent finite differences protect the derivative order and sign
        # at both an ordinary point and beyond the old hard clamp.
        s, n = 1, 2
        aux = (s = [s], ntr = [n], logchoose = [log(2.0)])
        for eta in (0.4, 8.0)
            h = 1e-4
            f(x) = BinomialLaplace._laplace_value(Val(:binomial), aux, 1, x)
            d1(x) = BinomialLaplace._laplace_d1(Val(:binomial), aux, 1, x)
            d2(x) = BinomialLaplace._laplace_d2(Val(:binomial), aux, 1, x)
            terms = BinomialLaplace._laplace_v123(Val(:binomial), aux, 1, eta)
            @test isapprox((f(eta + h) - f(eta - h)) / (2h), terms[2]; atol = 1e-9, rtol = 1e-8)
            @test isapprox((d1(eta + h) - d1(eta - h)) / (2h), terms[3]; atol = 1e-9, rtol = 1e-8)
            @test isapprox((d2(eta + h) - d2(eta - h)) / (2h), terms[4]; atol = 1e-9, rtol = 1e-8)
        end
    end
end

function stable_binomial_cloglog_reference(s, n, logchoose, eta)
    x = exp(BigFloat(eta))
    A = x / expm1(x)
    A1 = A * (1 - x - A)
    A2 = A1 * (1 - x - 2A) - A * x
    value = -BigFloat(s) * log(-expm1(-x)) + BigFloat(n - s) * x -
            BigFloat(logchoose)
    d1 = -BigFloat(s) * A + BigFloat(n - s) * x
    d2 = -BigFloat(s) * A1 + BigFloat(n - s) * x
    d3 = -BigFloat(s) * A2 + BigFloat(n - s) * x
    return value, d1, d2, d3
end

@testset "Binomial cloglog Laplace derivatives use the requested link" begin
    setprecision(BigFloat, 256) do
        for (s, n) in ((0, 6), (1, 6), (6, 6)), eta in (-20.0, -12.0, -4.0, 0.0, 2.0, 3.0)
            logchoose = log(binomial(n, s))
            aux = (s = [s], ntr = [n], logchoose = [logchoose], link = :cloglog)
            expected = stable_binomial_cloglog_reference(s, n, logchoose, eta)
            actual = BinomialLaplace._laplace_v123(Val(:binomial), aux, 1, eta)

            @test isapprox(actual[1], Float64(expected[1]); atol = 1e-12, rtol = 1e-12)
            @test isapprox(actual[2], Float64(expected[2]); atol = 1e-12, rtol = 1e-11)
            @test isapprox(actual[3], Float64(expected[3]); atol = 1e-11, rtol = 1e-10)
            @test isapprox(actual[4], Float64(expected[4]); atol = 1e-10, rtol = 1e-9)
            @test BinomialLaplace._laplace_value(Val(:binomial), aux, 1, eta) ≈ actual[1]
            @test BinomialLaplace._laplace_d1(Val(:binomial), aux, 1, eta) ≈ actual[2]
            @test BinomialLaplace._laplace_d2(Val(:binomial), aux, 1, eta) ≈ actual[3]
            @test BinomialLaplace._laplace_d3(Val(:binomial), aux, 1, eta) ≈ actual[4]
            @test collect(BinomialLaplace._laplace_d12(Val(:binomial), aux, 1, eta)) ≈ collect(actual[2:3])
        end

        s, n = 2, 5
        aux = (s = [s], ntr = [n], logchoose = [log(binomial(n, s))], link = :cloglog)
        for eta in (-4.0, 0.4, 2.0)
            h = 1e-4
            f(x) = BinomialLaplace._laplace_value(Val(:binomial), aux, 1, x)
            d1(x) = BinomialLaplace._laplace_d1(Val(:binomial), aux, 1, x)
            d2(x) = BinomialLaplace._laplace_d2(Val(:binomial), aux, 1, x)
            terms = BinomialLaplace._laplace_v123(Val(:binomial), aux, 1, eta)
            @test isapprox((f(eta + h) - f(eta - h)) / (2h), terms[2]; atol = 1e-7, rtol = 1e-6)
            @test isapprox((d1(eta + h) - d1(eta - h)) / (2h), terms[3]; atol = 1e-7, rtol = 1e-6)
            @test isapprox((d2(eta + h) - d2(eta - h)) / (2h), terms[4]; atol = 1e-6, rtol = 1e-5)
        end
    end
end
