using Base: Float64, @var, current_logger
using Gurobi, Cbc, Clp, JuMP, JSON, Printf, Base, DataStructures, UnitCommitmentSTO, GZip, LinearAlgebra
import Base: getindex, time
import UnitCommitmentSTO: Formulation, KnuOstWat2018, MorLatRam2013, ShiftFactorsFormulation

function val_mod(x)
    if abs(value(x))<1e-4
        return  0.0
    end
    return round(abs(value(x)), digits=6)
end    


function compute_total_payment(model, conf_key, payment_values, cost_values)
    

    instance = model[:instance]
    push!(payment_values, conf_key => OrderedDict{Any, Any}())
    push!(payment_values[conf_key], "generator payments" => OrderedDict{Any, Any}())
    for g in instance.units
        push!(payment_values[conf_key]["generator payments"], g.name => OrderedDict())
        gen_pay = payment_values[conf_key]["generator payments"][g.name]
        gen_cost = cost_values[conf_key]["generator costs"][g.name]
        gen_pay["hourly energy payments"] = OrderedDict(t =>
            value(model[:prod_above]["s1", g.name, t] +
            (model[:is_on][g.name, t] * g.min_power[t])) * 
            (-shadow_price(model[:eq_net_injection_def]["s1", g.bus.name, t])) for t in 1:instance.time)
        gen_pay["energy payment"] = sum(
            gen_pay["hourly energy payments"][t] for t in 1:instance.time)
        gen_pay["make whole payment"] =  max(gen_cost["total generator cost"] - gen_pay["energy payment"], 0)
        gen_pay["total payment"] = 
            gen_pay["energy payment"] + gen_pay["make whole payment"]
    end
end

function compute_total_load_payment(model, conf_key, load_pay)
    

    instance = model[:instance]
    push!(load_pay, conf_key => OrderedDict{Any, Any}())
    push!(load_pay[conf_key], "total load_payment" => sum(
        (b.scenarios[1].load[t] - value(model[:curtail]["s1", b.name,t])) *
            -shadow_price(model[:eq_net_injection_def]["s1", b.name, t])
            for t in 1:instance.time for b in instance.buses))
end


function compute_lmps(model, conf_key, lmp_values)


    instance = model[:instance]
    push!(lmp_values, conf_key => OrderedDict{Any, Any}())
    for bus in instance.buses
        push!(lmp_values[conf_key], bus.name => OrderedDict{Any, Any}())
        for t in 1:instance.time
            push!(lmp_values[conf_key][bus.name], t=>-shadow_price(model[:eq_net_injection_def]["s1", bus.name, t]) * instance.time_multiplier)
        end
    end
end


function compute_total_curtailment(model, conf_key, curtailment_values)

    instance  = model[:instance]
    push!(curtailment_values, conf_key => OrderedDict{Any, Any}())
    for bus in instance.buses
        for t in 1:instance.time
            push!(curtailment_values[conf_key], [bus.name, t]=>value(model[:curtail]["s1", bus.name,t]) / instance.time_multiplier)
        end
    end
end


function compute_total_cost(model, conf_key, cost_values)


    instance  = model[:instance]
    tf = 1/(instance.time_multiplier)
    sf = 1/instance.nscenarios
    T = instance.time
    push!(cost_values, conf_key => OrderedDict{Any, Any}())
    push!(cost_values[conf_key], "objective value sum" => objective_value(model))
    push!(cost_values[conf_key], "generator costs" => OrderedDict{Any, Any}())
    gen_cost = cost_values[conf_key]["generator costs"]
    for g in instance.units
        K = length(g.cost_segments)
        S = length(g.startup_categories)
        push!(gen_cost, g.name => OrderedDict())
        gen_cost[g.name]["variable production cost"] = sum(
            tf*sc.probability*value(model[:segprod][sc.name, g.name, t, k])*
            g.cost_segments[k].cost[t] 
            for k in 1:K for t in 1:T for sc in instance.buses[1].scenarios)
        gen_cost[g.name]["fixed commitment cost"] = sum(
            tf*sc.probability*value(model[:is_on][g.name, t])*g.min_power_cost[t] for
            t in 1:T for sc in instance.buses[1].scenarios)
        gen_cost[g.name]["startup cost"] = sum(
            sf*value(model[:startup][g.name, t, s])*
            g.startup_categories[s].cost 
            for s in 1:S for t in 1:T) 
        gen_cost[g.name]["total generator cost"] = gen_cost[g.name]["variable production cost"] +  
            gen_cost[g.name]["fixed commitment cost"] +  gen_cost[g.name]["startup cost"]
    end
    cost_values[conf_key]["total cost of generators"] = 
        sum(gen_cost[g.name]["total generator cost"] for g in instance.units)
    cost_values[conf_key]["load curtailment cost"] = sum(
        tf*sc.probability*value(model[:curtail][sc.name, b.name, t])*
        instance.power_balance_penalty[t] 
        for t in 1:T for b in instance.buses for sc in instance.buses[1].scenarios)
    cost_values[conf_key]["reserve shortfall penalty"] = sum(
        tf*sc.probability*instance.shortfall_penalty[t]*
        value(model[:reserve_shortfall][sc.name, t])
        for t in 1:T for sc in instance.buses[1].scenarios)
    cost_values[conf_key]["line overflow penalty"] = sum(
        sc.probability*value(model[:overflow][sc.name, lm.name, t])*
        lm.flow_limit_penalty[t]
        for t in 1:T for lm in instance.lines for sc in instance.buses[1].scenarios)
    cost_values[conf_key]["total cost"] = cost_values[conf_key]["load curtailment cost"] + 
        cost_values[conf_key]["reserve shortfall penalty"] + 
        cost_values[conf_key]["line overflow penalty"] +
        sum(cost_values[conf_key]["generator costs"][g.name]["total generator cost"] for g in instance.units)
end


function compute_oos_scenario_results(model, conf_key, payment_values, load_pay, lmp_values, curtailment_values, cost_values)
    vals = OrderedDict(v=> val_mod(v) for v in all_variables(model) if is_binary(v))
    for (v, val) in vals
        fix(v, val)
    end
    relax_integrality(model)
    JuMP.optimize!(model)
    compute_total_cost(model, conf_key, cost_values)
    compute_lmps(model, conf_key, lmp_values)
    compute_total_payment(model, conf_key, payment_values, cost_values)
    compute_total_load_payment(model, conf_key, load_pay)
    compute_total_curtailment(model, conf_key, curtailment_values)    
end