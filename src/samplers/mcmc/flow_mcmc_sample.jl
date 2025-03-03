
# function bat_sample_impl(
#     target::BATMeasure,
#     algorithm::TransformedMCMCSampling,
#     context::BATContext
# )
#     if(algorithm.adaptive_transform isa CustomTransform)
#         println("Using ensembles to sample with flows!")
#         return bat_sample_impl_ensemble(target,algorithm,context)
#     end

#     density, trafo = transform_and_unshape(algorithm.pre_transform, target, context)

#     init = mcmc_init!(
#         algorithm,
#         density,
#         algorithm.nchains,
#         apply_trafo_to_init(trafo, algorithm.init),
#         algorithm.tuning_alg,
#         algorithm.nonzero_weights,
#         algorithm.store_burnin ? algorithm.callback : nop_func,
#         context
#     )

#     @unpack chains, tuners, temperers = init

#     burnin_outputs_coll = if algorithm.store_burnin
#         DensitySampleVector(first(chains))
#     else
#         nothing
#     end

#     mcmc_burnin!(
#        burnin_outputs_coll,
#        chains,
#        tuners,
#        temperers,
#        algorithm.burnin,
#        algorithm.convergence,
#        algorithm.strict,
#        algorithm.nonzero_weights,
#        algorithm.store_burnin ? algorithm.callback : nop_func
#     )

#     # sampling
#     run_sampling  = _run_sample_impl(
#         density,
#         algorithm,
#         chains,
#     )
#     samples_trafo, generator = run_sampling.result_trafo, run_sampling.generator

#     # prepend burnin samples to output
#     if algorithm.store_burnin
#         burnin_samples_trafo = varshape(density).(burnin_outputs_coll)
#         append!(burnin_samples_trafo, samples_trafo)
#         samples_trafo = burnin_samples_trafo
#     end

#     samples_notrafo = inverse(trafo).(samples_trafo)
    

#     (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
# end


function bat_sample_impl_ensemble(
    target::BATMeasure,
    algorithm::TransformedMCMC,
    context::BATContext
)  

    density, pre_trafo = transform_and_unshape(algorithm.pretransform, convert(BATMeasure, target), context)
    vs = varshape(target)

    init = mcmc_init!(
        algorithm,
        density,
        algorithm.nchains,
        apply_trafo_to_init(pre_trafo, algorithm.init),
        algorithm.transform_tuning,
        algorithm.nonzero_weights,
        algorithm.store_burnin ? algorithm.callback : nop_func,
        context
    )

    @unpack ensembles, tuners, temperers = init

    burnin_outputs_coll = if algorithm.store_burnin
        DensitySampleVector(first(ensembles))
    else
        nothing
    end

    # burnin and tuning  # @TODO: Hier wird noch kein ensemble BurnIn gemacht !!!!!!!!!!!!!!!
   # mcmc_burnin!(
   #     burnin_outputs_coll,
   #     ensembles,
   #     tuners,
   #     temperers,
   #     algorithm.burnin,
   #     algorithm.convergence,
   #     algorithm.strict,
   #     algorithm.nonzero_weights,
   #     algorithm.store_burnin ? algorithm.callback : nop_func
   # )


    # sampling
    run_sampling = _run_sample_impl_ensemble(
        density,
        algorithm,
        ensembles,
        tuners,
        temperers
    )

    samples_trafo, generator, flow = run_sampling.result_trafo, run_sampling.generator, run_sampling.flow

    # prepend burnin samples to output
    if algorithm.store_burnin
        burnin_samples_trafo = burnin_outputs_coll
        prepend!(samples_trafo, burnin_samples_trafo)
    end

    global g_state_post_algorithm = (pre_trafo, samples_trafo, vs, density, algorithm, ensembles, tuners, temperers, target, context)

    samples_notrafo = inverse(pre_trafo).(samples_trafo)
    #samples_notrafo = vs.(inverse(pre_trafo).(samples_trafo))

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = pre_trafo, generator = TransformedMCMCSampleGenerator(ensembles, algorithm), flow = flow)
end


#=
function _bat_sample_continue(
    target::BATMeasure,
    generator::TransformedMCMCSampleGenerator,
    ;description::AbstractString = "MCMC iterate"
)
    @unpack algorithm, ensembles = generator
    density_notrafo = convert(AbstractMeasureOrDensity, target)
    density, trafo = transform_and_unshape(algorithm.pre_transform, density_notrafo)

    run_sampling = _run_sample_impl(density, algorithm, ensembles, description=description)

    samples_trafo, generator = run_sampling.result_trafo, run_sampling.generator

    samples_notrafo = inverse(trafo).(samples_trafo)

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = TransformedMCMCSampleGenerator(ensembles, algorithm))
end
=#

function _run_sample_impl(
    density::BATMeasure,
    algorithm::TransformedMCMC,
    chains::AbstractVector{<:MCMCIterator},
    ;description::AbstractString = "MCMC iterate"
)

    next_cycle!.(chains) 

    #progress_meter = ProgressMeter.Progress(algorithm.nchains*algorithm.nsteps, desc=description, barlen=80-length(description), dt=0.1)

    # tuners are set to 'NoOpTuner' for the sampling phase
    for i in 1:algorithm.nsteps
        transformed_mcmc_iterate!(
            chains,
            get_tuner.(Ref(TransformedMCMCNoOpTuning()),chains),
            get_temperer.(Ref(TransformedNoTransformedMCMCTempering()), chains),
            max_nsteps = algorithm.nsteps, #TODO: maxtime
            nonzero_weights = algorithm.nonzero_weights,
            #callback = (kwargs...) -> let pm=progress_meter; ProgressMeter.next!(pm) ; end,
        )
    end

    #ProgressMeter.finish!(progress_meter)


    output = reduce(vcat, getproperty.(chains, :samples))
    samples_trafo = varshape(density).(output)


    (result_trafo = samples_trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end

function _run_sample_impl_ensemble(
    density::BATMeasure,
    algorithm::TransformedMCMC,
    ensembles::AbstractVector{<:MCMCIterator},
    tuner,
    temperer,
    ;description::AbstractString = "MCMC iterate"
)

    next_cycle!.(ensembles) 

    #progress_meter = ProgressMeter.Progress(algorithm.nchains*algorithm.nsteps, desc=description, barlen=80-length(description), dt=0.1)

    # tuners are set to 'NoOpTuner' for the sampling phase
    println("Start Sampling phase")
    for i in 1:algorithm.nsteps
        transformed_mcmc_iterate!(
            ensembles,
            tuner,#get_tuner.(Ref(TransformedMCMCNoOpTuning()),ensembles),
            temperer, #get_temperer.(Ref(TransformedNoTransformedMCMCTempering()), ensembles),
            max_nsteps = algorithm.nsteps, #TODO: maxtime
            nonzero_weights = algorithm.nonzero_weights,
            #callback = (kwargs...) -> let pm=progress_meter; ProgressMeter.next!(pm) ; end,
        )
    end
    #ProgressMeter.finish!(progress_meter)

    output = copy(ensembles[1].states_x[1])
    for ensemble in ensembles
        for i in 1:length(ensemble.states_x)
            append!(output,ensemble.states_x[i])
        end
    end

    samples_trafo = varshape(density).(output[1:end])

    (result_trafo = samples_trafo, generator = TransformedMCMCSampleGenerator(ensembles, algorithm),flow = ensembles[1].f)
end
#
#    output = reduce(vcat, getfield.(ensembles, :states_x))
#
#    (result_trafo = output, generator = TransformedMCMCSampleGenerator(ensembles, algorithm))
#end