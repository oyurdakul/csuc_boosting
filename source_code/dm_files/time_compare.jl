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

basedir = homedir()
imp_dir = "$basedir/flexibility_repo/cso"
inp_file = "$imp_dir/test_scen300_weighted/stats.json"
inp_json = JSON.parse(open(inp_file, "r"), dicttype = () -> DefaultOrderedDict(nothing))
total_time=0
for (key,vals) in inp_json
    total_time+=inp_json[key]["time"]
end
print("Total computation time is $(total_time) seconds")