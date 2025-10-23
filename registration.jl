
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

#============================================#
# storage1.ris.wustl.edu\holy\Active\tom\vno_recordings\081425\highK_furine_ringers.imagine
# kim503@compute1-client-1.ris.wustl.edu:/storage1/fs1/holy/Active/tom/vno_recordings/081425/highK_furine_ringers.imagine
using AxisArrays, NRRD, FileIO, ImagineFormat, FixedPointNumbers

tomdir = "/storage1/fs1/holy/Active/tom/vno_recordings/081425"
mydir = "/storage1/fs1/holy/Active/daewoo/work/julia/RegisterWorker"
img = load(raw"/storage1/fs1/holy/Active/tom/vno_recordings/081425/highK_furine_ringers.imagine")

sizex, sizel, sizez, sizet = size(img.data)
axx = AxisArrays.Axis{:x}(1:sizex)
axl = AxisArrays.Axis{:l}(1:sizel)
axz = AxisArrays.Axis{:z}(1:sizez)
axt = AxisArrays.Axis{:time}(1:sizet)
header = NRRD.headerinfo(N0f16, (axx, axl, axz, axt))
header["datafile"] = joinpath(tomdir, "highK_furine_ringers.cam")
open(joinpath(mydir,"test.nhdr"),"w") do io # write header
    write(io,magic(format"NRRD"))
    NRRD.write_header(io,"0004",header)
end

#==================== RegisterDriver thread test ========================#
$ julia -t 16 # 16 threads julia
using BlockRegistration                                # just needed for this demo
brdir = dirname(dirname(pathof(BlockRegistration)))    # directory of BlockRegistration
include(joinpath(brdir, "test", "gen2d.jl"));          # defines `img`

using StaticArrays, JLD, ImageAxes, Test
using RegisterCore, RegisterWorkerApertures, RegisterWorkerShell, RegisterDriver, Unitful
using Base.Threads

gridsize = (3, 3)
shift_amplitude = 10
tax = ImageAxes.timeaxis(img)
nframes = length(tax)
nhalf = nframes ÷ 2
fixed = img[tax(nhalf+1)]
mxshift = (3*shift_amplitude, 3*shift_amplitude)
nodes = map(d->range(1, stop=size(fixed,d), length=gridsize[d]), (1,2))
pp = PreprocessSNF(0.1, [2,2], [10,10])

# fixed λ (with saving to file)
λ = 0.001
fn_pp = joinpath(tempdir(), "apertured_pp.jld")
algorithms = map(tid->Apertures(pp(fixed), nodes, mxshift, λ, pp, tid=tid),threadids()) # each thread has its own algorithm instance
mm_package_loader(algorithms)
mons = monitor(algorithms, (),
               Dict(:u => Array{SVector{2,Float64}}(undef, gridsize),
                    :warped => Array{Float64}(undef, size(fixed)),
                    :warped0 => Array{Float64}(undef, size(fixed)),
                    :mismatch => 0.0))
@time RegisterDriver.driver(fn_pp, algorithms, img, mons) # 493.674 ms (163503 allocations: 186.36 MiB)
           # 0.825767 seconds (180.50 k allocations: 213.857 MiB, 5.37% gc time, 6 lock conflicts)


#==================== RegisterDriver distributed test ========================#
include(joinpath(workpath,"gen2d.jl")) # generate deformed test image frames with cameraman image

using Distributed, SharedArrays, StaticArrays, JLD, ImageAxes, ImageView, Test
using RegisterCore, RegisterMismatch

aperturedprocs = addprocs(2)
@everywhere using RegisterWorkerApertures, RegisterWorkerShell, RegisterDriver, Unitful

# imshow(img)
gridsize = (3, 3)
shift_amplitude = 10
tax = ImageAxes.timeaxis(img)
nframes = length(tax)
nhalf = nframes ÷ 2
fixed = img[tax(nhalf+1)]
mxshift = (3*shift_amplitude, 3*shift_amplitude)
nodes = map(d->range(1, stop=size(fixed,d), length=gridsize[d]), (1,2))
pp = PreprocessSNF(0.1, [2,2], [10,10])

# fixed λ (with saving to file)
λ = 0.001
fn_pp = joinpath(tempdir(), "apertured_pp.jld")
sfixed = SharedArray{Float64}(size(fixed))
sfixed .= pp(fixed)
algorithms = Apertures[Apertures(sfixed, nodes, mxshift, λ, pp; pid=p) for p in aperturedprocs] # sfixed is shared not duplicated
algorithms2 = Apertures[Apertures(pp(fixed), nodes, mxshift, λ, pp; pid=p) for p in aperturedprocs] # fixed is duplicated
mm_package_loader(algorithms) # load mismatch packages on worker processes (RegisterMismatch or RegisterMismatchCuda)
mons = monitor(algorithms,
               (),
               Dict(:u => Array{SVector{2,Float64}}(undef, gridsize),
                    :warped => Array{Float64}(undef, size(fixed)),
                    :warped0 => Array{Float64}(undef, size(fixed)),
                    :mismatch => 0.0))
driver(fn_pp, algorithms, img, mons)

nfailures = 0
jldopen(fn_pp) do f
    global nfailures
    mm = read(f["mismatch"])
    @test all(mm .> 0)
    warped = read(f["warped"])
    for i = 1:nimages(img)
        r0 = RegisterCore.ratio(mismatch0(pp(fixed), pp(img[tax(i)])),0)
        r1 = RegisterCore.ratio(mismatch0(pp(fixed), warped[:,:,i]), 0)
        nfailures += r0 <= r1
    end
    warped0 = read(f["warped0"])
    for i = 1:nimages(img)
        r0 = RegisterCore.ratio(mismatch0(fixed, img[tax(i)]),0)
        r1 = RegisterCore.ratio(mismatch0(fixed, warped0[:,:,i]), 0)
        nfailures += r0 <= r1
    end
end
@test nfailures <= 2

# auto λrange (only one frame with no saving to file)
λrange = (1e-6,10)
alg = Apertures(pp(fixed), nodes, mxshift, λrange, pp) # no distributed computing
mm_package_loader(alg)
mon = monitor(alg, (), Dict(:λs=>0, :datapenalty=>0, :λ=>0, :u=>0, :warped0 => Array{Float64}(undef, size(fixed))))
mon = driver(alg, img[tax(1)], mon)
datapenalty = mon[:datapenalty]
@test !all(mon[:warped0] .== 0)

#============= CUDA test ===================#
using ImageMagick
using Distributed, SharedArrays, JLD, Test
using ImageCore, ImageAxes, ImageFiltering, TestImages
using StaticArrays, Interpolations
using RegisterCore, RegisterDeformation, RegisterMismatchCommon
using AxisArrays: AxisArray
using CUDA

aperturedprocs = addprocs(2)
# @everywhere using Pkg; Pkg.develop("RegisterWorkerApertures"); Pkg.develop("RegisterMismatchCuda"); Pkg.add("RegisterDriver")
@everywhere using RegisterWorkerApertures, RegisterDriver, RegisterMismatchCuda, Unitful


CUDA.allowscalar(false)

gridsize = (3, 3)
shift_amplitude = 10
tax = ImageAxes.timeaxis(img)
nframes = length(tax)
nhalf = nframes ÷ 2
fixed = img[tax(nhalf+1)]
mxshift = (3*shift_amplitude, 3*shift_amplitude)
nodes = map(d->range(1, stop=size(fixed,d), length=gridsize[d]), (1,2))
pp = PreprocessSNF(0.1, [2,2], [10,10])

λ = 0.001
fn_pp = joinpath(tempdir(), "apertured_pp_cuda.jld")
sfixed = SharedArray{Float32}(size(fixed))
sfixed .= pp(fixed)
algorithms = Apertures[Apertures(sfixed, nodes, mxshift, λ, pp; pid=p, dev=0) for p in aperturedprocs] # sfixed is shared not duplicated
#algorithms[2].dev = 0
mm_package_loader(algorithms) # load mismatch packages on worker processes (RegisterMismatch or RegisterMismatchCuda)
mons = monitor(algorithms,
               (),
               Dict(:u => Array{SVector{2,Float64}}(undef, gridsize),
                    :warped => Array{Float64}(undef, size(fixed)),
                    :warped0 => Array{Float64}(undef, size(fixed)),
                    :mismatch => 0.0))
driver(fn_pp, algorithms, img, mons)

nfailures = 0
jldopen(fn_pp) do f
    global nfailures
    mm = read(f["mismatch"])
    @test all(mm .> 0)
    warped = read(f["warped"])
    for i = 1:nimages(img)
        r0 = RegisterCore.ratio(mismatch0(pp(fixed), pp(img[tax(i)])),0)
        r1 = RegisterCore.ratio(mismatch0(pp(fixed), warped[:,:,i]), 0)
        nfailures += r0 <= r1
    end
    warped0 = read(f["warped0"])
    for i = 1:nimages(img)
        r0 = RegisterCore.ratio(mismatch0(fixed, img[tax(i)]),0)
        r1 = RegisterCore.ratio(mismatch0(fixed, warped0[:,:,i]), 0)
        nfailures += r0 <= r1
    end
end
@test nfailures <= 2

#=============== memory usage =====================#
Sys.total_memory() / 1e9 # 16.9362944   # 약 16.9 GB system RAM
Sys.free_memory() / 1e9 # 8.2036736    # 약 8.2 GB 사용 가능

memalloc = @allocated driver(fn_pp, algorithms, img, mons)
memalloc/1e6 # 30.197424 MB

@allocated ppfixed = pp(fixed) # 6.3MB (including function pp running)
ppfixedmem = Base.summarysize(ppfixed) # (only count array object(pointer and dimension) in julia heap + julia heap에 있는 포인터 참조 데이터까지 포함)
ppfixedmem/1e3 # 679.016KB
sizeof(eltype(ppfixed)) * length(ppfixed)/1e3 # 678.976KB 

sfixed412 = SharedArray{eltype(ppfixed)}(size(ppfixed)) # datasize = 678.976KB 
sfixed412mem = Base.summarysize(sfixed412) # (only count sharedarray object(pointer and dimension) in julia heap but not count data in the shared memory area)
sfixed412mem/1e3 # 679.629KB (metadata size is big)
sfixed1000 = SharedArray{eltype(ppfixed)}((1000,1000)) # datasize = 4MB
sfixed1000mem = Base.summarysize(sfixed1000) # (only count sharedarray object(pointer and dimension + 내부 Array wrapper()) in julia heap but not count data in the shared memory area)
sfixed1000mem/1e3 # 4000.653KB (metadata size is big)

mem = @allocated algorithms = Apertures[Apertures(pp(fixed), nodes, mxshift, λ, pp; pid=p) for p in aperturedprocs] # sfixed is shared not duplicated
mem/1e6 # 5.13KB (only count julia heap)
Base.summarysize(algorithms)/1e3 # 1360.128KB (pp(fixed) are duplicated and also duplicated in each worker)

mem = @allocated algorithms = Apertures[Apertures(ppfixed, nodes, mxshift, λ, pp; pid=p) for p in aperturedprocs] # sfixed is shared not duplicated
mem/1e6 # 5.13KB (only count julia heap)
Base.summarysize(algorithms)/1e3 # 681.112KB (ppfixed are shared but duplicated in each worker)

sharedmem = @allocated algorithms = Apertures[Apertures(sfixed412, nodes, mxshift, λ, pp; pid=p) for p in aperturedprocs] # sfixed is shared not duplicate
sharedmem/1e3 # 5.13KB (only count julia heap)
Base.summarysize(algorithms)/1e3 # 681.725KB (sfixed412 are shared and also shared in each worker)

mm_package_loader(algorithms) # load mismatch packages on worker processes (RegisterMismatch or RegisterMismatchCuda)
mons = monitor(algorithms,
               (),
               Dict(:u => Array{SVector{2,Float64}}(undef, gridsize),
                    :warped => Array{Float64}(undef, size(fixed)),
                    :warped0 => Array{Float64}(undef, size(fixed)),
                    :mismatch => 0.0))
driver(fn_pp, algorithms, img, mons)


#================================#
using Distributed

rmprocs(waitfor=1.0)
wids = addprocs(2)

#@everywhere global dev = nothing  # 모든 worker에 dev 변수를 미리 만듦
# --- 1️⃣ 각 worker에 dev 변수 설정
@spawnat wids[1] global dev7 = "GPU0"
@spawnat wids[2] global dev7 = "GPU1"

# --- 2️⃣ pmap 사용
results = pmap(1:4) do i
    # 이 코드는 각 worker 프로세스에서 실행됨
    return (myid(), dev7, i)
    #workfunc(i,dev7)
end

println(results)


#================================#
rmprocs(workers())
wids = addprocs(2)

@everywhere const wids2 = $wids   # ← 마스터의 wids를 모든 worker에 복제

@everywhere function init_dev()
    if myid() == wids2[1]
        global dev0 = "GPU0"
    elseif myid() == wids2[2]
        global dev0 = "GPU1"
    end
end

@everywhere init_dev()

@everywhere function workfunc(i, dev_local)
    (myid(), dev_local, i)
end

results = pmap(1:4) do i
    workfunc(i,dev0) # dev6는 각 worker에서 이미 정의되어 있음 하지만 master에 같은 이름이 정의되어 있으면 그 변수를 사용
end # UndefVarError: `dev0` not defined

dev5 = "CPU"
results = pmap(1:4) do i
    workfunc(i,dev5)
end # [(118, "CPU", 1), (117, "CPU", 2), (117, "CPU", 3), (118, "CPU", 4)]

#================================#
rmprocs(workers())
wids = addprocs(2)

@everywhere function workfunc(i, dev_local)
    (myid(), dev_local, i)
end

function mm_load(devs::Vector)
    wpids = map(t->t[1], devs)
    aindices = Dict(wpids[1]=>1, wpids[2]=>2)
    for wid in wpids
        #@spawnat wid eval(:(global dev6 = nothing))
        @spawnat wid global dev9 = devs[aindices[wid]][2]
    end
    results = pmap(1:4) do i
        workfunc(i,dev9) # dev9는 각 worker에서 이미 정의되어 있음 하지만 master에 같은 이름이 정의되어 있으면 그 변수를 사용
    end
end

devs = [(wids[1], "GPU0"), (wids[2],"GPU1")]
mm_load(devs) # ok [(119, "GPU0", 1), (120, "GPU1", 2), (117, "GPU0", 3), (118, "GPU1", 4)]


#================================#
using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\RegisterWorker"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/RegisterWorker"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
include(joinpath(workpath,"gen2d.jl")) # generate deformed test image frames with cameraman image

using Distributed, SharedArrays, StaticArrays, JLD, ImageAxes, ImageView, Test

rmprocs(workers())
wids = addprocs(2)

@everywhere module TestModule
    using Distributed

    function workfunc(i, dev_local)
        j = i*i
        (myid(), dev_local, j)
    end

    function mm_load(devs::Vector)
        wpids = map(t->t[1], devs)
        aindices = Dict(wpids[1]=>1, wpids[2]=>2)
        for wid in wpids
            #@spawnat wid eval(:(global dev6 = nothing))
            @spawnat wid global dev1 = devs[aindices[wid]][2]
        end
        results = pmap(1:4) do i
            workfunc(i, dev1) # dev11는 각 worker에서 이미 정의되어 있음 하지만 master에 같은 이름이 정의되어 있으면 그 변수를 사용
        end
    end
end

@everywhere using Main.TestModule

devs = [(wids[1], "GPU0"), (wids[2],"GPU1")]
results = TestModule.mm_load(devs) 
println(results) # ok [(130, "GPU1", 1), (129, "GPU0", 4), (130, "GPU1", 9), (129, "GPU0", 16)]

#================================#
using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\RegisterWorker"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/RegisterWorker"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")
include(joinpath(workpath,"gen2d.jl")) # generate deformed test image frames with cameraman image

using Distributed

rmprocs(workers(),waitfor=1.0)
wids = addprocs(2)
@everywhere using RegisterDriver # RegisterDriver is not loaded if the previous one is same as this
                                 # if you change some code in RegisterDriver and rerun from this line, it will work 
                                 # If you reallocate workers, you need to reopen julia 
devs = [(wids[1], "GPU1"), (wids[2],"GPU2")]
results = RegisterDriver.mm_load(devs) 
println(results) # ok [(130, "GPU1", 1), (129, "GPU0", 4), (130, "GPU1", 9), (129, "GPU0", 16)]

pool = CachingPool([myid()])  # master Task만 포함
pmap(pool, 1:2) do _
    println(myid())
    return nothing
end

#========== RIS Storage management ==========#
# check disk usage
df -Ph /storage1/fs1/holy/Active
df -Ph /storage1/fs1/holy/Archive
 
# ip Active to Archive
nohup bash -c 'for dir in /storage1/fs1/holy/Active/chantal/*/; do
    dirname=$(basename "$dir")
    echo "Would compress: $dir → /storage1/fs1/holy/Archive/chantal/${dirname}.tar.gz"
    tar -czf "/storage1/fs1/holy/Archive/chantal/${dirname}.tar.gz" -C /storage1/fs1/holy/Active/chantal "$dirname"
done' > /storage1/fs1/holy/Archive/chantal/archive_log.txt 2>&1 &

cat /storage1/fs1/holy/Archive/chantal/archive_log.txt

tar -tzf yourfile.tar.gz # check compression only with file name without decompression

# to check process list
ps aux | grep nohup
ps aux | grep 'tar -czf /storage1/fs1/holy'
