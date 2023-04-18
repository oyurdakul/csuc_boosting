using Base: Float64, @var, current_logger, Ordered
using Gurobi
using Cbc
using Clp
using JuMP
using JSON
using PyCall
using Printf
import Base
using JSON: print
using GZip
import Base: getindex, time
using DataStructures
import MathOptInterface
using LinearAlgebra
using Statistics
function without_keys(dict, key_s)
    return Dict(key => dict[key] for key in keys(dict) if key ∉ key_s)
end
function selected_keys(dict, scenario_number)
    temp = Dict(key => dict[key] for key in keys(dict) if occursin("weight", key))
    return Dict(key => temp[key] for key in keys(temp) if occursin("snum_$(scenario_number)", key))
end
Xi_vals = Dict(10 => 4.0, 20 => 10.0, 50 => 4.0, 100 => 4.0, 200 => 4.0, 300 => 4.0, 365 => 10.0)
basedir = homedir()
imp_dir = "$basedir/flexibility_repo/cso/test_300"
scenario_numbers = [300]
types = ["weighted"]
dict_overall = Dict()
for scenario_number in scenario_numbers
    dict_scen = Dict()
    for type in types
        dict_type = Dict()
        inp_file = "$imp_dir/snum_$(scenario_number)/cost_results.json"
        inp_json = JSON.parse(open(inp_file, "r"), dicttype = () -> DefaultOrderedDict(nothing))
        inp_time = "$imp_dir/stats.json"
        inp_json_time = JSON.parse(open(inp_time, "r"), dicttype = () -> DefaultOrderedDict(nothing))
        if type == "weighted" || type == "point"
            inp_str = inp_json[type]["rf"]["configuration costs"]
        else
            inp_str = inp_json[type][type]["configuration costs"]
        end
        mean_val_cost = 0
        standard_deviation_cost = 0
        for (key_d,dict) in inp_str
            dict_mod = without_keys(dict, ["sum"])
            mean_val_cost = mean(values(dict_mod))
            standard_deviation_cost = std(values(dict_mod))
        end
        push!(dict_type, "mean cost"=>mean_val_cost)
        push!(dict_type, "upper level cost"=>mean_val_cost + (standard_deviation_cost)/10)
        push!(dict_type, "lower level cost"=>mean_val_cost - (standard_deviation_cost)/10)
        if type == "weighted"
            time_dict_mod = selected_keys(inp_json_time, scenario_number)
            i = 0
            dict_times = Dict()
            for (key, dict_time) in time_dict_mod
                i += 1
                push!(dict_times, i=>dict_time["time"])
            end
            mean_val_time = mean(values(dict_times))
            standard_deviation_time = std(values(dict_times))
            push!(dict_type, "mean time"=>mean_val_time)
            push!(dict_type, "upper level time"=>mean_val_time + (standard_deviation_time)/10)
            push!(dict_type, "lower level time"=>mean_val_time - (standard_deviation_time)/10)
        end
        push!(dict_scen, type=>dict_type)
    end
    push!(dict_overall, scenario_number=>dict_scen)
end
open("$imp_dir/plots_300.json","w") do f
    JSON.print(f, dict_overall)
end