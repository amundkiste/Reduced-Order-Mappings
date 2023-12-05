@everywhere using LinearAlgebra
@everywhere using Distributions
@everywhere using Random
@everywhere using SparseArrays
@everywhere using NPZ

@everywhere include("./Navier-Stokes-Force-Sol.jl");

function Data_Generate()
    #viscosity_inv = 1
    N_data = 10000
    #ν = 1.0/viscosity_inv                                      # viscosity
    N, L = 64, 2*pi                                 # resolution and domain size 
    N_t = 10000;                                     # time step
    T = 10.0;                                        # final time

    d=4.0
    τ=3.0
    # The forcing has N_θ terms
    N_θ = 100
    seq_pairs = Compute_Seq_Pairs(100)

    Random.seed!(42);
    θf = rand(Normal(0,1), N_data, N_θ)
    v = rand(Uniform(0,1), N_data)
    curl_f = zeros(N, N, N_data)
    for i = 1:N_data
    	# 2*
        curl_f[:,:, i] .= generate_ω0(L, N, θf[i,:], seq_pairs, d, τ)
    end

    θω = rand(Normal(0,1), N_θ)
    ω0 = generate_ω0(L, N, θω, seq_pairs, d, τ)

    # Define caller function
    @everywhere g_(x) = 
        NS_Solver(x[1], $ω0, x[2];N_t = $N_t, T = $T)

    
    params = [(curl_f[:, :, i], v[i]) for i in 1:N_data]
    @everywhere params = $params
    ω_tuple = pmap(g_, params) # Outer dim is params iterator

    ω_field = zeros(N, N, N_data)
    for i = 1:N_data
        ω_field[:,:, i] = ω_tuple[i]
    end

    npzwrite("viscosity_$(N_data).npy",  v)
    npzwrite("theta_$(N_data).npy",  θf)
    npzwrite("omega_$(N_data).npy",  ω_field)
    npzwrite("curl_f_$(N_data).npy", curl_f)
    npzwrite("omega0_$(N_data).npy", ω0)

end

Data_Generate()