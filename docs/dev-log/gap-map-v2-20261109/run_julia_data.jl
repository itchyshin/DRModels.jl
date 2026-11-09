using DRModels, DelimitedFiles
const DD = get(ENV, "FUZZ_DATA", "data")
const FACS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
const d = let (m, h) = readdlm(joinpath(DD, "data_v2.csv"), ',', Any; header = true)
    h = String.(vec(h)); cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(h)
        c = m[:, j]
        v = if n in FACS
            any(x -> x == "NA", c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] : String.(c)
        elseif any(x -> x == "NA", c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        elseif all(x -> x isa Integer, c)
            Int.(c)
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    (; cols...)
end
# R spellings that Julia's @formula spells differently: interaction `a:b` is `a & b`, and `I(x^2)` has no
# counterpart, so it is rewritten to the function term `sq(x)`. Applied to every formula string.
sq(x) = x .^ 2
jl_spell(s) = replace(replace(s, "I(x1^2)" => "sq(x1)"), r"(\w+):(\w+)" => s"\1 & \2")
F(s) = eval(Meta.parse("@formula(" * jl_spell(s) * ")"))

