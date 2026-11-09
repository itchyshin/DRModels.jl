# Load the fuzz runner's cell builder without running its main loop.
src = read(joinpath(@__DIR__, "harness", "run_julia.jl"), String)
i = findfirst("open(outfile, \"a\")", src)
body = src[1:first(i)-1]
body = replace(body, "cells_file, idfile, start, outfile = ARGS[1], ARGS[2], parse(Int, ARGS[3]), ARGS[4]" =>
  "cells_file = joinpath(@__DIR__, \"harness\", \"cells_run.tsv\"); idfile = joinpath(@__DIR__, \"harness\", \"ids54.txt\"); start = 1; outfile = \"/dev/null\"")
include_string(Main, body)
