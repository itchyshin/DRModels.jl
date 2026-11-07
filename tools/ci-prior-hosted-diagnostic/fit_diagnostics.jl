println("DIAGNOSTIC_FIT_BEGIN")
println("converged=", slope_fit.converged, " iterations=", slope_fit.iterations)
println("theta=", repr(slope_fit.theta))
println("theta_bits=", bitstring.(slope_fit.theta))
println("loglik=", repr(slope_fit.loglik), " control=", repr(slope_fit.control_info))
println("data_sha256=", bytes2hex(SHA.sha256(repr(data))))
if slope_fit.nll !== nothing
    try
        println("objective=", repr(slope_fit.nll(slope_fit.theta)))
        diagnostic_gradient = ForwardDiff.gradient(slope_fit.nll, slope_fit.theta)
        println("gradient=", repr(diagnostic_gradient), " maxabs=", maximum(abs, diagnostic_gradient))
    catch diagnostic_error
        println("DIAGNOSTIC_GRADIENT_ERROR=", sprint(showerror, diagnostic_error))
    end
end
println("DIAGNOSTIC_FIT_END")
flush(stdout)
