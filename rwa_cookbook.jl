
using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\RegisterWorker"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    if  gethostname() == "creed"
        workpath="/home/daewoo/work/julia/RegisterWorker"
        datapath="/home/daewoo/work/Data"
    else
        workpath=ENV["MYSTORAGE"]*"/work/julia/RegisterWorker"
        datapath=ENV["MYSTORAGE"]*"/work/Data"
    end
end
cd(workpath); Pkg.activate(".")

using Distributed, SharedArrays, StaticArrays, RegisterWorkerShell, JLD, FileIO, ImageCore, ImageTransformations, ImageAxes

wpids = addprocs(10)

@everywhere using RegisterWorkerApertures, RegisterDriver, Unitful, ImagineFormat


# Some physical units we may need (from the Unitful package)
const μm = u"μm"  # micrometers
const s  = u"s"   # seconds

#### Load image on the master process
# Normally this might be `img = load("myimagefile")`, but this is a demo
# img = load(raw"/media/tom/TOM_DATA/081425/highK_furine_ringers.imagine").data
img = load(raw"/storage1/fs1/holy/Active/tom/vno_recordings/081425/highK_furine_ringers.imagine").data

#### Choose the fixed image and set up the parameters (this is similar to BlockRegistration)
fixedidx = (nimages(img)+1) ÷ 2  # ÷ can be obtained with "\div[TAB]"
fixed = img[time=fixedidx]

# Important: you should manually inspect fixed to make sure there are
# no anomalies. Do not proceed to the next step until you have done this.

# Choose the maximum amount of movement you want to allow (set this by visual inspection)
mxshift = (30, 30, 30)  # 30 pixels along each spatial axis for a 2d+time image
# Pick a grid size for your registration. Finer grids allow more
# "detail" in the deformation, but also have more parameters and
# therefore require higher SNR data.
gridsize = (3, 3, 3)   # it's fine to choose something anisotropic

# Choose volume regularization penalty. See the help for BlockRegistration.
λ = 1e-5

# Compute the nodes from the image and grid
nodes = map(ImageAxes.axes(fixed), gridsize) do ax, g
    range(first(ax), stop=last(ax), length=g)
end

#### Set up the workers, the monitor, and run it via the driver
# Create the worker algorithm structures. We assign one per worker process.
# @allocated sfixed = SharedArray{eltype(fixed)}(size(fixed))
# @allocated sfixed .= fixed
algorithm = [Apertures(fixed, nodes, mxshift, λ; pid=wpids[i], correctbias=false) for i = 1:length(wpids)]

# Set up the "monitor" which aggregates the results from the workers
@allocated mon = monitor(algorithm, (), Dict{Symbol,Any}(:u=>ArrayDecl(Array{SVector{3,Float64},3}, gridsize)))

# Load the appropriate mismatch package
mm_package_loader(algorithm)

# Define the output file and run the job
fileout = "results2.register"
@time driver(fileout, algorithm, img, mon) # 10 workers : 3261.116310 seconds (27.60 M allocations: 1.400 GiB, 0.02% gc time, 2 lock conflicts, 0.31% compilation time: <1% of which was recompilation)

# Append important extra information to the file
jldopen(fileout, "r+") do io
    write(io, "fixedidx", fixedidx)
    write(io, "nodes", nodes)
end