using Base: Float64, @var, current_logger
using Gurobi, Cbc, Clp, JuMP, JSON, Printf, Base, DataStructures, UnitCommitmentSTO, GZip, LinearAlgebra
import Base: getindex, time
import UnitCommitmentSTO: Formulation, KnuOstWat2018, MorLatRam2013, ShiftFactorsFormulation

include("compute_oos_scenario_results.jl")

function first_stage(ml_algorithm, scenario_number, oos_scenario_number, conf, conf_path, benchmark_method)
    
    instance = UnitCommitmentSTO.read("input_files/snum_$(scenario_number)/" *
        "oos_$(oos_scenario_number)/$(benchmark_method)/$(ml_algorithm)/$(conf_path).json",)
    model = UnitCommitmentSTO.build_model(
        instance = instance,
        optimizer = Gurobi.Optimizer,
        formulation = Formulation(
            
            )
    )
    time_stat = @timed UnitCommitmentSTO.optimize!(model)
    solution = UnitCommitmentSTO.solution(model)
    UnitCommitmentSTO.write("input_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/" *
        "$(benchmark_method)/$(ml_algorithm)/$(conf_path)_fs.json", solution)
    return time_stat
end

function second_stage(ml_algorithm, scenario_number, oos_scenario_number, conf, conf_path, 
        benchmark_method, gen_pay, load_pay, lmp_vals, cost_vals, curt_vals)
    
    instance = UnitCommitmentSTO.read("input_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/" * 
        "oos_$(oos_scenario_number).json",)
    model = UnitCommitmentSTO.build_model(
    instance = instance,
    optimizer = Gurobi.Optimizer,
    formulation = Formulation(
        )
    )
    mul = instance.time_multiplier
    if benchmark_method !== "ideal"
        inp_file = "input_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/$(benchmark_method)/" *
            "$(ml_algorithm)/$(conf_path)_fs.json"
        inp_json = JSON.parse(open(inp_file, "r"), dicttype = () -> DefaultOrderedDict(nothing))
        [@constraint(model, model[:is_on][g.name,t] == inp_json["Is on"][g.name][div(t-1,mul)+1]) 
            for g in instance.units for t in 1:instance.time]
    end
    time_stat = @timed UnitCommitmentSTO.optimize!(model)
    solution = UnitCommitmentSTO.solution(model)
    UnitCommitmentSTO.write("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/$(benchmark_method)/" *
        "$(ml_algorithm)/$(conf_path)_sol.json", solution)
    compute_oos_scenario_results(model, conf_path, gen_pay, load_pay, lmp_vals, curt_vals, cost_vals)

end