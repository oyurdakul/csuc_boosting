using Base: Float64, @var, current_logger
using Gurobi, Cbc, Clp, JuMP, JSON, Printf, Base, DataStructures, UnitCommitmentSTO, GZip, LinearAlgebra
import Base: getindex, time
import UnitCommitmentSTO: Formulation, KnuOstWat2018, MorLatRam2013, ShiftFactorsFormulation

function write_generator_payment_results(payment_results, benchmark_method, scenario_number,
    ml_algorithm, number_of_oos_scenarios)
    json_f = open("output_files/snum_$(scenario_number)/oos_1/$(benchmark_method)/" *
        "$(ml_algorithm)/payment_values.json", "r")
    dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
    configuration_keys = [key for key in keys(dict["detailed payment"])]
    generator_keys = [key for key in keys(dict["detailed payment"][configuration_keys[1]]["generator payments"])]
    for conf_key in configuration_keys
        push!(payment_results[benchmark_method][ml_algorithm]["configuration payments"], 
            conf_key => OrderedDict{Any,Any}())
        for gname in generator_keys
            payment_types = OrderedDict{Any, Any}(
                    "energy payment" =>0, 
                    "make whole payment"=>0, 
                    "total payment" =>0)
            for oos_scenario_number in 1:number_of_oos_scenarios
                json_f = open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/$(benchmark_method)/" *
                    "$(ml_algorithm)/payment_values.json", "r")
                dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
                gen_payment = dict["detailed payment"][conf_key]["generator payments"][gname]
                for payment_type in keys(payment_types)
                    payment_types[payment_type] += convert(Float64, gen_payment[payment_type])
                end
            end
            payment_results[benchmark_method][ml_algorithm]["configuration payments"][conf_key][gname] =
                payment_types
        end
        gen_pay = payment_results[benchmark_method][ml_algorithm]["configuration payments"][conf_key]
        gnames = [gname for gname in keys(gen_pay)]
        payment_type_keys = ["energy payment", "make whole payment", "total payment"]
        for payment_type in payment_type_keys
            push!(gen_pay, "aggregate $(payment_type)" => sum(convert(Float64, gen_pay[gname][payment_type]) for gname in gnames))
        end
    end
end

function write_cost_breakdowns(cost_results, benchmark_method, scenario_number,
    ml_algorithm, number_of_oos_scenarios)
    json_f = open("output_files/snum_$(scenario_number)/oos_1/$(benchmark_method)/" *
        "$(ml_algorithm)/cost_values.json", "r")
    dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
    configuration_keys = [key for key in keys(dict["detailed cost results"])]
    generator_keys = [key for key in keys(dict["detailed cost results"][configuration_keys[1]]["generator costs"])]
    for conf_key in configuration_keys
        push!(cost_results[benchmark_method][ml_algorithm]["configuration costs"], 
            conf_key => OrderedDict{Any,Any}("generator costs"=> OrderedDict{Any,Any}()))
        for gname in generator_keys
            gen_cost_types = OrderedDict{Any, Any}(
                    "variable production cost" =>0, 
                    "fixed commitment cost"=>0, 
                    "startup cost" =>0,
                    "total generator cost"=>0)
            for oos_scenario_number in 1:number_of_oos_scenarios
                json_f = open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/$(benchmark_method)/" *
                    "$(ml_algorithm)/cost_values.json", "r")
                dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
                ind_gen_cost = dict["detailed cost results"][conf_key]["generator costs"][gname]
                for cost_type in keys(gen_cost_types)
                    gen_cost_types[cost_type] += convert(Float64, ind_gen_cost[cost_type])
                end
            end
            cost_results[benchmark_method][ml_algorithm]["configuration costs"][conf_key]["generator costs"][gname] =
                gen_cost_types
        end
        gen_costs = cost_results[benchmark_method][ml_algorithm]["configuration costs"][conf_key]["generator costs"]
        gnames = [gname for gname in keys(gen_costs)]
        cost_type_keys = ["variable production cost", "fixed commitment cost", "startup cost", "total generator cost"]
        for cost_type in cost_type_keys
            push!(gen_costs, "aggregate $(cost_type)" => sum(convert(Float64, gen_costs[gname][cost_type]) for gname in gnames))
        end
        other_cost_types = OrderedDict{Any, Any}(
            "load curtailment cost" =>0, 
            "reserve shortfall penalty"=>0, 
            "line overflow penalty" =>0)
        for oos_scenario_number in 1:number_of_oos_scenarios
            json_f = open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/$(benchmark_method)/" *
                "$(ml_algorithm)/cost_values.json", "r")
            dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
            ind_other_cost = dict["detailed cost results"][conf_key]
            for cost_type in keys(other_cost_types)
                other_cost_types[cost_type] += convert(Float64, ind_other_cost[cost_type])
            end
        end
        conf_costs = cost_results[benchmark_method][ml_algorithm]["configuration costs"][conf_key]
        for cost_type in keys(other_cost_types)
            push!(conf_costs, "aggregate $(cost_type)" => other_cost_types[cost_type])
        end
        push!(conf_costs, "aggregate total cost" => (conf_costs["generator costs"]["aggregate total generator cost"] + 
            sum(other_cost_types[cost_type] for cost_type in keys(other_cost_types))))

    end
end

function write_lmp_results(lmp_results, benchmark_method, scenario_number,
    ml_algorithm, number_of_oos_scenarios)
    json_f = open("output_files/snum_$(scenario_number)/oos_1/$(benchmark_method)/" *
        "$(ml_algorithm)/lmp_values.json", "r")
    dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
    configuration_keys = keys(dict)
    for conf_key in configuration_keys
        lmp_result = lmp_results[benchmark_method][ml_algorithm]["configuration lmps"][conf_key]
        lmp_result["average lmps"] = OrderedDict{Any,Any}(
            bus_name => OrderedDict{Any,Any}(
            time_period_index => 0 for time_period_index in keys(lmp_result["oos values"][1][bus_name]))
            for bus_name in keys(lmp_result["oos values"][1])
        )
        for bus_name in keys(lmp_result["average lmps"])
            for time_period_index in keys(lmp_result["average lmps"][bus_name])
                for oos_scenario_number in 1:number_of_oos_scenarios
                    json_f = open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/$(benchmark_method)/" *
                        "$(ml_algorithm)/lmp_values.json", "r")
                    dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
                    lmp_result["average lmps"][bus_name][time_period_index] += dict[conf_key][bus_name][time_period_index]
                end
                lmp_result["average lmps"][bus_name][time_period_index] /= number_of_oos_scenarios 
            end
        end
        
    end
end

function collect_oos_results(scenario_number, benchmark_method, ml_algorithm, 
    result_type, result_dic, number_of_oos_scenarios)
    for oos_scenario_number in 1:number_of_oos_scenarios
        json_f = open("output_files/snum_$(scenario_number)/oos_$(oos_scenario_number)/$(benchmark_method)/" *
        "$(ml_algorithm)/$(result_type)_values.json", "r")
        dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
        if result_type != "lmp"
            push!(result_dic[benchmark_method][ml_algorithm]["oos $(result_type)s"], 
                oos_scenario_number => dict["total $(result_type)"])
        else
            push!(result_dic[benchmark_method][ml_algorithm]["oos $(result_type)s"], 
                oos_scenario_number => dict)
        end
    end
    
    json_f = open("output_files/snum_$(scenario_number)/oos_1/$(benchmark_method)/" *
        "$(ml_algorithm)/payment_values.json", "r")
    dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
    conf_keys = keys(dict["total payment"])

    # write results for each configuration 
    push!(result_dic[benchmark_method][ml_algorithm], "configuration $(result_type)s" => 
    OrderedDict{Any,Any}(conf_key => 
        OrderedDict{Any,Any}("oos values" =>
        OrderedDict{Any,Any}(oos_scenario_number =>
            result_dic[benchmark_method][ml_algorithm]["oos $(result_type)s"][oos_scenario_number][conf_key] 
                for oos_scenario_number in 1:number_of_oos_scenarios)) 
                    for conf_key in conf_keys))
    # compute the sum of the oos results for each configuration 
    
    if result_type !== "lmp"
        for conf_key in conf_keys
            push!(result_dic[benchmark_method][ml_algorithm]["configuration $(result_type)s"][conf_key]["oos values"], "sum" =>  
                sum(
                    convert(Float64, result_dic[benchmark_method][ml_algorithm]["oos $(result_type)s"][oos_scenario_number][conf_key])
                    for oos_scenario_number in 1:number_of_oos_scenarios))
        end
    end
    # write individual generator payment results
    if result_type == "payment"
        write_generator_payment_results(result_dic, benchmark_method, scenario_number,
            ml_algorithm, number_of_oos_scenarios)
    elseif result_type == "cost"
        write_cost_breakdowns(result_dic, benchmark_method, scenario_number,
            ml_algorithm, number_of_oos_scenarios)
    elseif result_type == "lmp"
        write_lmp_results(result_dic, benchmark_method, scenario_number,
            ml_algorithm, number_of_oos_scenarios)
    end

    open("output_files/snum_$(scenario_number)/$(result_type)_results.json","w") do f
        JSON.print(f, result_dic, 4)
    end
end

function write_results(scenario_number, params, benchmark_methods)
    payment_results = OrderedDict{Any,Any}()
    load_payment_results = OrderedDict{Any,Any}()
    cost_results = OrderedDict{Any,Any}()
    curtailment_results = OrderedDict{Any,Any}()
    lmp_results=OrderedDict{Any,Any}()
    results = [ ("lmp", lmp_results), ("cost", cost_results), ("payment", payment_results), ("load_payment",
        load_payment_results), ("curtailment", curtailment_results)]
    for benchmark_method in benchmark_methods
        if benchmark_method === "weighted" || benchmark_method === "point"
            ml_algorithm_names = params["ml_algorithms"]
        else
            ml_algorithm_names = [benchmark_method]
        end
        for result in results
            push!(result[2], benchmark_method => OrderedDict{Any,Any}())
        end
        for ml_algorithm in ml_algorithm_names
            push!(payment_results[benchmark_method], ml_algorithm => OrderedDict{Any,Any}("oos payments" => OrderedDict{Any,Any}()))
            push!(lmp_results[benchmark_method], ml_algorithm => OrderedDict{Any,Any}("oos lmps" => OrderedDict{Any,Any}()))
            push!(cost_results[benchmark_method], ml_algorithm => OrderedDict{Any,Any}("oos costs" => OrderedDict{Any,Any}()))
            push!(curtailment_results[benchmark_method], ml_algorithm => OrderedDict{Any,Any}("oos curtailments" => OrderedDict{Any,Any}()))
            push!(load_payment_results[benchmark_method], ml_algorithm => OrderedDict{Any,Any}("oos load_payments" => OrderedDict{Any,Any}()))
            for result in results
                collect_oos_results(scenario_number, benchmark_method, ml_algorithm, 
                    result[1], result[2], params["number_of_oos_scenarios"])
            end
        end

    end

end