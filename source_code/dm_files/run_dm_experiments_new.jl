using Base: Float64, @var, current_logger
using Gurobi, Cbc, Clp, JuMP, JSON, Printf, Base, DataStructures, UnitCommitmentSTO, GZip, LinearAlgebra
import Base: getindex, time
import UnitCommitmentSTO: Formulation, KnuOstWat2018, MorLatRam2013, ShiftFactorsFormulation

include("two_stage_solution.jl")
include("write_results.jl")
include("retrieve_best_hyperparameters.jl")


function run_dm_experiments(params, benchmark_methods)
    stat_dict = Dict()
    for scenario_number in params["number_of_scenarios"]
            for oos_scenario_number in 1:params["number_of_oos_scenarios"]
                for benchmark_method in benchmark_methods
                    if benchmark_method === "weighted" || benchmark_method === "point"
                        ml_algorithms = params["ml_algorithms"]
                    else
                        ml_algorithms = [benchmark_method]
                    end
                    for ml_algorithm in ml_algorithms
                        mkpath("output_files/snum_$(scenario_number)/" *
                            "oos_$(oos_scenario_number)/$(benchmark_method)/$(ml_algorithm)/")
                        gen_pay_det = OrderedDict()
                        load_pay = OrderedDict()
                        cost_vals = OrderedDict() 
                        lmp_vals = OrderedDict()  
                        curt_vals_det = OrderedDict()
                        prepare_two_stage_problem(scenario_number, oos_scenario_number, benchmark_method, ml_algorithm, params,
                        gen_pay_det, load_pay, lmp_vals, cost_vals, curt_vals_det, stat_dict)
                        write_oos_results(scenario_number, oos_scenario_number, benchmark_method, ml_algorithm, gen_pay_det,
                        load_pay, cost_vals, lmp_vals, curt_vals_det)
                        
                    end
                end
            end
        write_results(scenario_number, params, benchmark_methods)
        retrieve_best_hyperparameters(scenario_number)
    end

    return stat_dict
end

function formulate_two_stage_problem(scenario_number, oos_scenario_number, benchmark_method, ml_algorithm,
    conf, conf_path, stat_key, gen_pay_det, load_pay, lmp_vals, cost_vals, curt_vals_det, stat_dict)
    if benchmark_method !== "ideal"
        time_val = first_stage(ml_algorithm, scenario_number, oos_scenario_number, conf, conf_path, benchmark_method)
        stat_dict[stat_key] = time_val
    end
    second_stage(ml_algorithm, scenario_number, oos_scenario_number, conf, conf_path, benchmark_method, gen_pay_det, 
        load_pay, lmp_vals, cost_vals, curt_vals_det)
end

function prepare_two_stage_problem(scenario_number, oos_scenario_number, benchmark_method, ml_algorithm, params,
    gen_pay_det, load_pay, lmp_vals, cost_vals, curt_vals_det, stat_dict)
    if benchmark_method === "weighted" || benchmark_method === "point"
        for n_est in params[ml_algorithm]["n_estimators"]
            for m_dep in params[ml_algorithm]["max_depth"]
                for l_rat in params[ml_algorithm]["learning_rate"]
                    for m_spl in params[ml_algorithm]["min_split_loss"]
                        for m_sam in params[ml_algorithm]["min_samples_split"]
                            for m_fea in params[ml_algorithm]["max_features"]
                                if benchmark_method == "point"
                                    conf = Dict("n_estimators" => n_est, "Xi" => "xi", 
                                        "max_depth" => m_dep, "learning_rate" => l_rat, "min_split_loss" => m_spl, 
                                        "min_samples_split" => m_sam, "max_features" => m_fea )
                                    conf_path = "n_est_$(n_est)_Xi_xi_max_depth_$(m_dep)_lear_rate_$(l_rat)" *
                                        "_min_sp_l_$(m_spl)_min_samples_split_$(m_sam)_max_features_$(m_fea)"
                                    stat_key = "point_model_$(ml_algorithm)_snum_$(scenario_number)" * 
                                    "_n_est_$(n_est)_max_depth_$(m_dep)_lear_rate_$(l_rat)_min_sp_l_" * 
                                    "$(m_spl)_min_samples_split_$(m_sam)_max_features_$(m_fea)_Xi_xi_oos_" * 
                                    "$(oos_scenario_number)"
                                    formulate_two_stage_problem(scenario_number, oos_scenario_number, benchmark_method, ml_algorithm,
                                    conf, conf_path, stat_key, gen_pay_det, load_pay, lmp_vals, cost_vals, curt_vals_det, stat_dict)
                                else benchmark_method == "weighted"
                                    for xi in params[ml_algorithm]["Xi"]
                                        conf = Dict("n_estimators" => n_est, "Xi" => xi, 
                                            "max_depth" => m_dep, "learning_rate" => l_rat, "min_split_loss" => 
                                            m_spl, "min_samples_split" => m_sam, "max_features" => m_fea )
                                        conf_path = "n_est_$(n_est)_Xi_$(xi)_max_depth_$(m_dep)_" * 
                                            "lear_rate_$(l_rat)_min_sp_l_$(m_spl)_min_samples_split_" *
                                            "$(m_sam)_max_features_$(m_fea)"
                                        stat_key = "weighted_model_$(ml_algorithm)_snum_$(scenario_number)" * 
                                            "_n_est_$(n_est)_max_depth_$(m_dep)_lear_rate_$(l_rat)_min_sp_l_" * 
                                            "$(m_spl)_min_samples_split_$(m_sam)_max_features_$(m_fea)_Xi_$(xi)_oos_" * 
                                            "$(oos_scenario_number)"
                                        formulate_two_stage_problem(scenario_number, oos_scenario_number, benchmark_method, ml_algorithm,
                                            conf, conf_path, stat_key, gen_pay_det, load_pay, lmp_vals, cost_vals, curt_vals_det, stat_dict)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    else 
        conf = conf_path = benchmark_method
        stat_key = "$(benchmark_method)_snum_$(scenario_number)_oos_$(oos_scenario_number)"
        formulate_two_stage_problem(scenario_number, oos_scenario_number, benchmark_method, ml_algorithm,
                conf, conf_path, stat_key, gen_pay_det, load_pay, lmp_vals, cost_vals, curt_vals_det, stat_dict)
    end
end

function write_oos_results(scenario_number, oos_scenario_number, benchmark_method, ml_algorithm, gen_pay_det,
    load_pay, cost_vals, lmp_vals, curt_vals_det)

    # Cost results

    open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/" *
    "$(benchmark_method)/$(ml_algorithm)/cost_values.json","w") do f
        JSON.print(f, OrderedDict(
            "total cost" => OrderedDict(conf_key =>
                cost_vals[conf_key]["total cost"] for conf_key in keys(cost_vals)),
            "detailed cost results" => 
                cost_vals), 4)
    end

    ## Load payment results
    open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/" *
    "$(benchmark_method)/$(ml_algorithm)/load_payment_values.json","w") do f
        JSON.print(f, OrderedDict(
            "total load_payment" => OrderedDict(conf_key =>
                load_pay[conf_key]["total load_payment"] for conf_key in keys(load_pay)),
            "detailed load payment results" => 
                load_pay), 4)
    end

    ####
    
    open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/" *
        "$(benchmark_method)/$(ml_algorithm)/lmp_values.json","w") do f
        JSON.print(f, lmp_vals, 4)
    end

    #####

    gen_pay = OrderedDict()
    push!(gen_pay, "detailed payment"=>gen_pay_det)
    push!(gen_pay, "total payment" => 
    Dict(conf_name => sum(results["generator payments"][gname]["total payment"] for gname in keys(results["generator payments"])) 
        for (conf_name, results) in gen_pay_det))
    push!(gen_pay, "total make whole payments" => 
    Dict(conf_name => sum(results["generator payments"][gname]["make whole payment"] for gname in keys(results["generator payments"])) 
        for (conf_name, results) in gen_pay_det))
    push!(gen_pay, "total energy payments" => 
    Dict(conf_name => sum(results["generator payments"][gname]["energy payment"] for gname in keys(results["generator payments"])) 
        for (conf_name, results) in gen_pay_det))
    open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/" *
        "$(benchmark_method)/$(ml_algorithm)/payment_values.json","w") do f
        JSON.print(f, gen_pay, 4)
    end
    
    ####
    
    curt_vals = OrderedDict()
    push!(curt_vals, "detailed curtailment"=>curt_vals_det)
    temp_curt = OrderedDict()
    for (key, dict) in curt_vals_det
        push!(temp_curt, key => sum([val for (key2, val) in dict]))
    end
    push!(curt_vals, "total curtailment"=>temp_curt)
    open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/" *
        "$(benchmark_method)/$(ml_algorithm)/curtailment_values.json","w") do f
        JSON.print(f, curt_vals, 4)
    end                                          
                    
end