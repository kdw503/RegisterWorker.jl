#===== RegisterWorkerAperture ======#
alg = Apertures(fixed, nodes, maxshift, λ, [preprocess=identity]; kwargs...)
creates a worker-object for performing "apertured" (blocked)
registration.  
fixed : reference image
nodes : grid of apertures
maxshift : represents the largest shift
λ : coefficient for the deformation penalty (higher values enforce a more affine-like deformation)
preprocess : allows you to apply a transformation (e.g.,filtering) to the `moving` images before registration; `fixed` should
already have any such transformations applied.

λ may be specified as a (λmin, λmax) tuple, in which case the "best" λ is chosen for you automatically via the algorithm described in `auto_λ`.  If you monitor the variable datapenalty, you can inspect the quality of the sigmoid used to choose λ.                                

pp = img -> imfilter_gaussian(img, [3, 3])
fixed = pp(fixed0)
# We'll use a 5x7 grid of apertures
nodes = (linspace(1, size(fixed,1), 5), linspace(1, size(fixed,2), 7))
maxshift = (30,30)
λrange = (1e-6, 100)
alg = Apertures(fixed, nodes, maxshift, λrange, pp)

# Monitor the datapenalty, the chosen value of λ, the deformation u, and also collect the corrected (warped) image. By asking for
# :warped0, we apply the warping to the unfiltered moving image (:warped would refer to the filtered moving image).
# We pre-allocate space for :warped0 to illustrate a trick for reducing the overhead of communication between worker and driver
# processes, even though this example uses just a single process. 
mon = monitor(alg, (), Dict(:λs=>0, :datapenalty=>0, :λ=>0, :u=>0, :warped0 => Array(Float64, size(fixed))))

# Run the algorithm
mon = driver(alg, moving0, mon)

# Plot the datapenalty and see how sigmoidal it is. Assumes you're
# `using Immerse`.
λs = mon[:λs]
datapenalty = mon[:datapenalty]
plot(x=λs, y=datapenalty, xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)


mutable struct Apertures{A<:AbstractArray,T,K,N} <: AbstractWorker
    fixed::A
    nodes::NTuple{N,K}
    maxshift::NTuple{N,Int}
    affinepenalty::AffinePenalty{T,N}
    overlap::NTuple{N,Int}
    λrange::Union{T,Tuple{T,T}}
    thresh::T
    preprocess  # likely of type PreprocessSNF, but could be a function
    normalization::Symbol
    correctbias::Bool
    workerpid::Int
    dev::Int
    cuda_objects::Dict{Symbol,Any}
end

Apertures(fixed, )
load_mm_package(dev)
init!(algorithm::Apertures)
close!(algorithm::Apertures)
worker(algorithm::Apertures, img, tindex, mon)
    moving0 = getindex_t(img, tindex)
    if use_cuda
        aperture_centers = aperture_grid(map(d->size(img,d),cs), gridsize)
        mms = allocate_mmarrays(eltype(cms), gridsize, algorithm.maxshift)
        mismatch_apertures!(mms, d_fixed, d_moving, aperture_centers, cms; normalization=algorithm.normalization)
    else
        aperture_centers = aperture_grid(map(d->size(img,d),cs), gridsize)
        aperture_width = default_aperture_width(algorithm.fixed, gridsize, algorithm.overlap)
        mms = mismatch_apertures(algorithm.fixed, moving, aperture_centers, aperture_width, )
    end
    for i = 1:length(mms)
        E0[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt=false)
    end
    mmis = interpolate_mm!(mms)
    λrange = algorithm.λrange
    if isa(λrange, Number)
        ϕ, mismatch = RegisterOptimize.fixed_λ(cs, Qs, algorithm.nodes, algorithm.affinepenalty, mmis)
    else
        ϕ, mismatch, λ, λs, dp, quality = RegisterOptimize.auto_λ(cs, Qs, algorithm.nodes, algorithm.affinepenalty, mmis, λrange)
        monitor!(mon, :λs, λs)
        monitor!(mon, :datapenalty, dp)
    end


#=== RegisterWorkerShell ======#
monitor(algorithm::AbstractWorker,)

"""
`init!(algorithm)` performs any necessary initialization prior to
beginning a registration sequence using algorithm `algorithm`. The
default action is to return `nothing`. If you require initialization,
specialize this function for your `AbstractWorker` subtype.
"""
init!(algorithm::AbstractWorker, args...) = nothing
init!(rr::RemoteChannel, args...) = init!(fetch(rr), args...)

close!(algorithm::AbstractWorker, args...) = nothing
close!(rr::RemoteChannel, args...) = close!(fetch(rr), args...)

worker(algorithm::AbstractWorker, img, tindex, mon) = error("Worker modules must define `worker`")
worker(rr::RemoteChannel, img, tindex, mon) = worker(fetch(rr), img, tindex, mon)

load_mm_package(dev, args...) = nothing
load_mm_package(rr::RemoteChannel, args...) = load_mm_package(fetch(rr), args...)



#=== RegisterDriver ======#
`driver(outfile, algorithm, img, mon)` performs registration of the
image(s) in `img` according to the algorithm selected by
`algorithm`. `algorithm` is either a single instance, or for parallel
computation a vector of instances, of `AbstractWorker` types.  See the
`RegisterWorkerShell` module for more details.

Results are saved in `outfile` according to the information in `mon`.
`mon` is a `Dict`, or for parallel computation a vector of `Dicts` of
the same length as `algorithm`.  The data saved correspond to the keys
(always `Symbol`s) in `mon`, and the values are used for communication
between the worker(s) and the driver.  The usual way to set up `mon`
is like this:

```
    algorithm = RegisterRigid(fixed, params...)   # An AbstractWorker algorithm
    mon = monitor(algorithm, (:tform,:mismatch))  # List of variables to record
```

The list of symbols, taken from the field names of `RegisterRigid`,
specifies the pieces of information to be communicated back to the
driver process for saving and/or display to the user.  It's also
possible to request local variables in the worker, as long as the
worker has been written to look for such settings:

```
    # <in the worker algorithm>
    monitor_copy!(mon, :extra, extra)
```

function driver(outfile::AbstractString, algorithm::Vector, img, mon::Vector)
    nworkers = length(algorithm)
    use_workerprocs = nworkers > 1 || workerpid(algorithm[1]) != myid()
    rralgorithm = Array{RemoteChannel}(undef, nworkers)
    if use_workerprocs
        # Push the algorithm objects to the worker processes. This elminates
        # per-iteration serialization penalties, and ensures that any
        # initalization state is retained.
        for i = 1:nworkers
            alg = algorithm[i]
            rralgorithm[i] = put!(RemoteChannel(workerpid(alg)), alg)
        end
        # Perform any needed worker initialization
        @sync for i = 1:nworkers
            p = workerpid(algorithm[i])
            @async remotecall_fetch(init!, p, rralgorithm[i])
        end
    else
        init!(algorithm[1])
    end
    n = nimages(img)
    fs = FormatSpec("0$(ndigits(n))d")  # group names of unpackable objects
    jldopen(outfile, "w") do file
        dsets = Dict{Symbol,Any}()
        firstsave = SharedArray{Bool}(1)
        firstsave[1] = true
        have_unpackable = SharedArray{Bool}(1)
        have_unpackable[1] = false
        # Run the jobs
        nextidx = 0
        getnextidx() = nextidx += 1
        writing_mutex = RemoteChannel()
        @sync begin
            for i = 1:nworkers
                alg = algorithm[i]
                @async begin
                    while (idx = getnextidx()) <= n
                        remotecall_fetch(println, workerpid(alg), "Worker ", workerpid(alg), " is working on ", idx)
                        # See https://github.com/JuliaLang/julia/issues/22139
                        tmp = remotecall_fetch(worker, workerpid(alg), rralgorithm[i], img, idx, mon[i])
                        copy_all_but_shared!(mon[i], tmp)
                        # Save the results
                        put!(writing_mutex, true)  # grab the lock
                        try
                            local g
                            if firstsave[]
                                firstsave[] = false
                                have_unpackable[] = initialize_jld!(dsets, file, mon[i], fs, n)
                            end
                            if fetch(have_unpackable[])
                                g = file[string("stack", fmt(fs, idx))]
                            end
                            for (k,v) in mon[i]
                                g[string(k)] = v
                            end
                        finally
                            take!(writing_mutex)   # release the lock
                        end
                    end
                end
            end
        end
    end
end

driver(outfile::AbstractString, algorithm::AbstractWorker, img, mon::Dict) = driver(outfile, [algorithm], img, [mon])

"""
`mon = driver(algorithm, img, mon)` performs registration on a single
image, returning the results in `mon`.
"""
function driver(algorithm::AbstractWorker, img, mon::Dict)
    nimages(img) == 1 || error("With multiple images, you must store results to a file")
    init!(algorithm)
    worker(algorithm, img, 1, mon)
    close!(algorithm)
    mon
end

mm_package_loader(algorithm::AbstractWorker) = mm_package_loader([algorithm])
function mm_package_loader(algorithms::Vector)
    nworkers = length(algorithms)
    use_workerprocs = nworkers > 1 || workerpid(algorithms[1]) != myid()
    rrdev = Array{RemoteChannel}(undef, nworkers)
    if use_workerprocs
        for i = 1:nworkers
            dev = algorithms[i].dev
            rrdev[i] = put!(RemoteChannel(workerpid(algorithms[i])), dev)
        end
        @sync for i = 1:nworkers
            p = workerpid(algorithms[i])
            @async remotecall_fetch(load_mm_package, p, rrdev[i])
        end
    else
        load_mm_package(algorithms[1].dev)
    end
    nothing
end

initialize_jld!



#========== Distributed computing =============#
> julia -p 4   # starts Julia with 4 worker processes

# or

using Distributed
addprocs(4)  # adds 4 workers

#Check how many processes are running:
nprocs()     # total number of processes (including master)
workers()

# Broadcast code to all workers with:
@everywhere using LinearAlgebra   # loads module on all processes

# Remote execution (futures):
fut = @spawn sum(rand(10^7))   # run on a worker
fetch(fut)                     # get the result

# Remote calls:
remotecall_fetch(+, 2, 10, 20)  # run 10 + 20 on worker 2

# Parallel loops
# For data parallelism:
@distributed (+) for i in 1:1_000_000
    rand()
end

# Shared arrays
# For shared-memory parallelism (single machine):
using SharedArrays
A = SharedArray{Float64}(10_000)
@distributed for i in 1:length(A)
    A[i] = i^2
end

# Higher-level parallel mapping functions
# pmap → good for tasks that take varying time (load-balanced mapping):
pmap(sqrt, 1:10^6)

# Threads.@threads (multi-threading, not multi-process):
Threads.@threads for i in 1:10^6
    # thread-based parallelism
end

#================ GPT code suggestion ================#
using Distributed

# add n CPU worker processes
n = 4
addprocs(n)

# Switch to GPU if requested
const USE_GPU = true   # set false for CPU

@everywhere begin
    using LinearAlgebra
    using Statistics
    # Load CUDA only if GPU mode is enabled
    if $USE_GPU
        using CUDA
    end

    # Dummy registration function
    # (replace this with your actual registration method)
    function register(refimg, img; gridsize=(10,10))
        # Here just return block-averaged displacement grid as an example
        m, n = size(refimg)
        gx, gy = gridsize
        out = zeros(Float32, gx, gy)

        blockx = div(m, gx)
        blocky = div(n, gy)

        for i in 1:gx, j in 1:gy
            x = (i-1)*blockx+1 : i*blockx
            y = (j-1)*blocky+1 : j*blocky
            out[i,j] = mean(refimg[x,y]) - mean(img[x,y])
        end
        return out
    end

    # Wrapper to allow GPU arrays
    function register_frame(refimg, frame; gridsize=(10,10))
        if $USE_GPU
            refimg_d = cu(refimg)
            frame_d  = cu(frame)
            res_d = register(refimg_d, frame_d; gridsize=gridsize)
            return Array(res_d)   # bring back to CPU
        else
            return register(refimg, frame; gridsize=gridsize)
        end
    end
end

# Example dataset
imgfrm = rand(Float32, 50, 50, 100)
refimg = imgfrm[:,:,50]
gridsize = (10,10)

# Parallel GPU/CPU processing
results = pmap(i -> register_frame(refimg, imgfrm[:,:,i]; gridsize=gridsize),
               1:size(imgfrm,3))

# Collect results into array
dv = Array{Float32}(undef, 5, 5, size(imgfrm,3))
for (i,res) in enumerate(results)
    dv[:,:,i] = res
end


#===================================================#
방법 1: @spawnat 사용
for i in 1:nworkers
    alg = algorithm[i]
    @spawnat workers()[i] begin
        for iter = 1:niterations
            # alg 사용
        end
    end
end

방법 2: pmap (입력 데이터 병렬 매핑)
function run_alg(alg)
    for iter = 1:niterations
        # alg 사용
    end
end

pmap(run_alg, algorithm)

@distributed for iter ... 은 반복 분배 방식 → 워커별로 다른 alg 유지하려는 목적에는 부적합
대신 @spawnat 또는 pmap 을 써야 워커별 다른 알고리즘 객체를 독립적으로 반복 실행 가능


* 예시 코드 (@spawnat 방식)
results = Vector{Any}(undef, nworkers)  # 결과 저장용 벡터

for (i, pid) in enumerate(workers())
    alg = algorithm[i]
    results[i] = @spawnat pid begin
        local_result = []
        for iter = 1:niterations
            push!(local_result, (pid, iter, some_computation(alg)))
        end
        local_result
    end
end

# fetch로 결과 모으기
final_results = [fetch(r) for r in results]

* 더 간단한 방법 (pmap 방식)
function run_alg(alg, pid)
    local_result = []
    for iter = 1:niterations
        push!(local_result, (pid, iter, some_computation(alg)))
    end
    return local_result
end

final_results = pmap((a,p)->run_alg(a,p), algorithm, workers()) # 고정 매핑

* 알고리즘 개수 > 워커 개수
알고리즘이 더 많음 → 워커에 작업을 분배해서 여러 번 돌려야 함
이때는 작업 큐 방식을 써야 효율적
가장 간단한 방법: pmap
final_results = pmap(run_alg, algorithm) # 자동 분배. 워커의 수는 julia -p 3 myscript.jl 또는 addprocs(3) 로 지정

pmap은 algorithm 벡터를 워커들에게 자동 분배
워커 수보다 알고리즘이 많으면 → 워커가 끝날 때마다 새로운 alg 할당
즉, 자동으로 작업 큐처럼 동작

#======================= GPT suggestion (마스터에서 파일로 저장) =============================#
using Distributed, JLD2

@everywhere function run_alg(alg, img, idx)
    # 실제 계산 로직 (워커에서 실행)
    tmp = worker(alg, img, idx)   # 원래 코드의 worker 함수
    return (idx=idx, result=tmp)
end

function process_all(algorithms, img, outfile)
    n = nimages(img)
    indices = 1:n

    # pmap으로 작업을 분산 실행
    results = pmap(indices) do idx
        # idx를 워커에 할당할 때, 해당 워커에 alg를 전달
        # 간단하게는 round-robin 매칭
        alg = algorithms[(idx - 1) % length(algorithms) + 1]
        run_alg(alg, img, idx)
    end

    # 마스터 프로세스에서 결과를 순서대로 저장
    jldopen(outfile, "w") do file
        for r in results
            idx = r.idx
            g = file["stack$(lpad(idx, ndigits(n), '0'))"]
            for (k,v) in r.result
                g[string(k)] = v
            end
        end
    end
end

#============================================#
# A : 1000 by1000 big size matrix
# B : 1000 by1000 by10000 big size array
# C # 100 by 100 by 10000 array 
# for i 1:10000
#    C[:,:,i] = do_something(A,B[:,:,i]) # A, B are only read not written
# end
#
# I have n workers, want to do distributed computing using pmap

addprocs(n)  # Add n workers
@everywhere using YourModule  # or whatever module defines `do_something`

# Ensure A and B are available on all workers
@everywhere A_remote = $A
@everywhere B_remote = $B

# Parallel map over indices
C_results = pmap(1:10000) do i
    do_something(A_remote, B_remote[:,:,i])
end

# Convert results to array
C = reshape(hcat(C_results...), size(B))  # shape: 100×100×10000

#================ 여러게의 워커가 동일 GPU 나눠 사용 ============================#
using Distributed
addprocs(10)  # CPU 워커 10개 추가

@everywhere begin
    using CUDA
    using Random

    # 워커별 GPU 초기화
    function init_gpu(gpu_id::Int)
        CUDA.device!(gpu_id)
        CUDA.gc()  # GPU 메모리 정리
        println("Worker $(myid()) initialized on GPU $gpu_id")
    end

    # 워커별 GPU 작업
    function gpu_task(task_id::Int, N::Int)
        CUDA.device!(0)  # 모든 워커가 GPU 0 사용
        Random.seed!(task_id)

        # 워커별 독립 배열 생성 (큰 배열도 안전)
        x = CUDA.rand(Float32, N, N)
        # 간단한 연산 예시
        y = sum(x)  
        # GPU 메모리 정리 (옵션)
        CUDA.gc()
        return y
    end
end

# 워커 초기화
pmap(pid -> remotecall_fetch(init_gpu, pid, 0), workers())

# GPU 작업 분배
N = 2000  # 큰 배열 예시
task_ids = 1:10  # 10개 워커용 작업 ID
results = pmap(task_id -> gpu_task(task_id, N), task_ids)

println("Results: ", results)