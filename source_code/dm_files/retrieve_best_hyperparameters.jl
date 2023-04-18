using Base: Float64, @var, current_logger
using Gurobi, Cbc, Clp, JuMP, JSON, Printf, Base, DataStructures, UnitCommitmentSTO, GZip, LinearAlgebra
import Base: getindex, time
import UnitCommitmentSTO: Formulation, KnuOstWat2018, MorLatRam2013, ShiftFactorsFormulation

function retrieve_best_hyperparameters(scenario_number)
    result_types = ["cost", "payment"]
    for result_type in result_types 
        json_f = open("output_files/snum_$(scenario_number)/$(result_type)_results.json", "r")
        json_fs=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
        for (benchmark_method, ml_costs) in json_fs
            for (ml_algorithm, configuration_costs) in ml_costs
               
                best_configuration = reduce((x, y) -> configuration_costs["configuration $(result_type)s"][x]["aggregate total $(result_type)"] ≤ configuration_costs["configuration $(result_type)s"][y]["aggregate total $(result_type)"] ? x : y, keys(configuration_costs["configuration $(result_type)s"]))
                best_configuration_result = configuration_costs["configuration $(result_type)s"][best_configuration]["aggregate total $(result_type)"]
                push!(configuration_costs, "best configuration" =>OrderedDict())
                configuration_costs["best configuration"][best_configuration] = best_configuration_result
            end
        end
        open("output_files/snum_$(scenario_number)/$(result_type)_results.json","w") do f
            JSON.print(f, json_fs, 4)
        end
    end
    
end
