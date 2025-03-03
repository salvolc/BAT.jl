# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct TransformedMCMCEnsemblePoolInit <: MCMCInitAlgorithm

MCMC ensemble pool initialization strategy.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TransformedMCMCEnsemblePoolInit <: MCMCInitAlgorithm
    init_tries_per_walker::ClosedInterval{Int64} = ClosedInterval(4, 128)
    nsteps_init::Int64 = 100
    initval_alg::InitvalAlgorithm = InitFromTarget()
    nwalker::Int = 100
    use_mala::Bool = false
    tau::Float64 = 0.1
end

function TransformedMCMCEnsemblePoolInit(
    init_tries_per_walker::ClosedInterval{Int64},
    nsteps_init::Int64,
    initval_alg::InitvalAlgorithm
) TransformedMCMCEnsemblePoolInit(init_tries_per_walker=init_tries_per_walker, nsteps_init=nsteps_init, initval_alg=initval_alg)
end

export TransformedMCMCEnsemblePoolInit


function apply_trafo_to_init(trafo::Function, initalg::TransformedMCMCEnsemblePoolInit)
    TransformedMCMCEnsemblePoolInit(
    initalg.init_tries_per_walker,
    initalg.nsteps_init,
    apply_trafo_to_init(trafo, initalg.initval_alg)
    )
end


function _construct_ensemble(
    rngpart::RNGPartition,
    id::Integer,
    algorithm::TransformedMCMC,
    density::BATMeasure,
    initval_alg::InitvalAlgorithm,
    nwalker::Integer,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = nestedview(ElasticArray{Float64}(undef, totalndof(varshape(density)), 0))
    
    for i in 1:nwalker
        push!(v_init, bat_initval(density, initval_alg, new_context).result)
    end

    return TransformedMCMCEnsembleIterator(algorithm, density, Int32(id), v_init, new_context)

    # test=nestedview(ElasticArray{Float64}(undef, totalndof(density), 0))
    # while (length(test)<nwalker)
    #     t=BAT.mcmc_init!(BAT.TransformedMCMC(),
    #         density, 100,
    #         BAT.TransformedMCMCChainPoolInit(),
    #         BAT.TransformedMCMCNoOpTuning(), # TODO: part of algorithm? # MCMCTuner
    #         true,
    #         x->x,
    #         new_context)[1]
    #     chains = getproperty.(t, :samples)
    #     for c in chains
    #         push!(test,c.v[end])
    #     end
    # end
    # #samp = Matrix{Float64}(test[1:nwalker]')
# 
    # return TransformedMCMCEnsembleIterator(
    #     algorithm,
    #     density,
    #     Int32(id),
    #     test[1:nwalker],
    #     new_context,
    # )
end

_gen_ensembles(
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    algorithm::TransformedMCMC,
    density::BATMeasure,
    initval_alg::InitvalAlgorithm,
    nwalker::Int,
    context::BATContext
) = [_construct_ensemble(rngpart, id, algorithm, density, initval_alg, nwalker,context) for id in ids]


function mcmc_init!(
    algorithm::TransformedMCMC,
    density::BATMeasure,
    nensembles::Integer,
    init_alg::TransformedMCMCEnsemblePoolInit,
    tuning_alg::MCMCTransformTuning, # TODO: part of algorithm? # MCMCTuner
    nonzero_weights::Bool,
    callback::Function,
    context::BATContext
)
    @info "TransformedMCMCEnsemblePoolInit: trying to generate $nensembles viable MCMC states_x(s)."

    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_walker)
    max_ncandidates::Int = maximum(init_alg.init_tries_per_walker)

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC ensemble to determine ensemble, output and tuner types." #TODO: remove!

    dummy_context = deepcopy(context)
    dummy_init_state = nestedview(ElasticArray{Float64}(undef, totalndof(varshape(density)), 0))
    push!(dummy_init_state, bat_initval(density, InitFromTarget(), dummy_context).result)

    dummy_ensemble = TransformedMCMCEnsembleIterator(algorithm, density, Int32(1), dummy_init_state, dummy_context) 
    dummy_tuner = get_tuner(tuning_alg, dummy_ensemble)
    dummy_temperer = create_temperering_state(algorithm.tempering, density)

    states_x = similar([dummy_ensemble], 0)
    tuners = similar([dummy_tuner], 0)
    temperers = similar([dummy_temperer], 0)

    transformed_mcmc_iterate!(
        states_x, tuners, temperers
    )

    outputs = []

    nwalker::Int = algorithm.init.nwalker

    n = min(min_nviable, max_ncandidates - ncandidates)
    println("Generating $(n*nwalker) candidate MCMC walker(s) per ensemble.")

    new_ensembles = _gen_ensembles(rngpart, ncandidates .+ (one(Int64):nensembles), algorithm, density, initval_alg,nwalker*n,context)
    #new_tuners = get_tuner.(Ref(NoMCMCTransformTuningState()),new_ensembles) # NoOpTuner for BurnIn
    new_tuners = [NoMCMCTransformTuningState() for i in new_ensembles] # NoOpTuner for BurnIn
    new_temperers = fill(create_temperering_state(algorithm.tempering, density), size(new_ensembles,1))

    #next_cycle!.(new_ensembles)
    #tuning_init!.(new_tuners, new_ensembles, init_alg.nsteps_init) # TODO Probablty the right function to build flow inside BAT

    @debug "Testing $(length(new_ensembles)) candidate MCMC ensemble(s)."
    for i in 1:init_alg.nsteps_init
        transformed_mcmc_iterate!(
            new_ensembles, new_tuners, new_temperers
        )
    end

    for ensemble in new_ensembles
        mask = (ensemble.n_accepted .> 0.03*init_alg.nsteps_init)
        viable_walker = ensemble.states_x[end][mask]
        choosen = rand(1:length(viable_walker),nwalker)
        ensemble.states_x = [viable_walker[choosen]]
        ensemble.state_z = ensemble.state_z[mask][choosen]
        ensemble.n_accepted = zeros(Int64,nwalker)
        if(length(viable_walker)<nwalker)
            println("Found not enough good walker!") # TODO retry with more walkers to find enough good ones!
        end
    end

    new_tuners = get_tuner.(Ref(tuning_alg), new_ensembles)
    new_outputs = getproperty.(new_ensembles, :states_x)             

    return (ensembles = new_ensembles, tuners = new_tuners, temperers = new_temperers, outputs = new_outputs)
end
