push!(LOAD_PATH,"UnitCommitment/src/")
using Base: Float64, @var, current_logger, Ordered, Order, Float32
using Gurobi, Cbc, Clp, JuMP, JSON, Printf, Base, DataStructures, UnitCommitmentSTO, GZip, LinearAlgebra
import Base: getindex, time
import UnitCommitmentSTO: Formulation, KnuOstWat2018, MorLatRam2013, ShiftFactorsFormulation

include("run_dm_experiments_new.jl")


benchmark_methods = ["weighted", "point", "ideal", "naive"]
params = Dict()
open("params.json", "r") do f
    global params
    params=JSON.parse(f; dicttype=DataStructures.OrderedDict)
end

stat_dict = run_dm_experiments(params, benchmark_methods)

open("output_files/stats.json","w") do f
    JSON.print(f, stat_dict, 4)     
end                