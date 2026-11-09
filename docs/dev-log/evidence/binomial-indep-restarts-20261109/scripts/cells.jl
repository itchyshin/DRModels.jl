using DRModels, DelimitedFiles
function g12_data(path)
    m, h = readdlm(path, ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    (y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")), x3 = Float64.(col("x3")),
     z = Float64.(col("z")), g = String.(col("g")))
end
g12_form(q) = q == 2 ?
    bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)),
       @formula(sd(g, coef(intercept)) ~ z), @formula(sd(g, coef(x1)) ~ z)) :
    bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) + (0 + x2 | g)),
       @formula(sd(g, coef(intercept)) ~ z), @formula(sd(g, coef(x1)) ~ z), @formula(sd(g, coef(x2)) ~ z))
