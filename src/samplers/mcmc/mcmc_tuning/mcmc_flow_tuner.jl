# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using AdaptiveFlows


"""
    MCMCFlowTuning <: MCMCTransformTuning

Train the Normalizing Flow that is used in conjunction with a MCMC sampler.
"""
struct MCMCFlowTuning <: MCMCTransformTuning end
export MCMCFlowTuning



struct MCMCFlowTuner <: MCMCTransformTunerState 
    optimizer
    n_batches::Integer
    n_epochs::Integer
end
export MCMCFlowTuner

(tuning::MCMCFlowTuning)(chain::MCMCIterator) = MCMCFlowTuner(AdaptiveFlows.Adam(0.005), 2, 20)
get_tuner(tuning::MCMCFlowTuning, chain::MCMCIterator) = MCMCFlowTuner(AdaptiveFlows.Adam(0.005), 2, 20)


function MCMCFlowTuning(tuning::MCMCFlowTuning, chain::MCMCIterator)
    MCMCFlowTuner(AdaptiveFlows.Adam(0.005), 2, 20)
end


function tuning_init!(tuner::MCMCFlowTuner, chain::MCMCIterator, max_nsteps::Integer)
    chain.info = TransformedMCMCIteratorInfo(chain.info, tuned = true) # Add a counter for training epochs of a flow instead? 
    nothing
end

g_state_flow_optimization = (;)

function tune_mcmc_transform!!(
    tuner::MCMCFlowTuner, 
    flow::AdaptiveFlows.AbstractFlow,
    x::AbstractArray,
    target,
    context::BATContext
)   
    # TODO find better way to handle ElasticMatrices
    global g_state_flow_optimization = (x, flow, tuner)

    target_logpdf = x -> BAT.checked_logdensityof(target).(x)
    flow_new = AdaptiveFlows.optimize_flow(nestedview(Matrix(flatview(x))), flow, tuner.optimizer, loss=AdaptiveFlows.negll_flow, nbatches = tuner.n_batches, 
                                                        nepochs = tuner.n_epochs, shuffle_samples = true, logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf))

    eta = tuner.optimizer.eta*0.98
    if (eta < 5f-5)
        eta = 5f-5
    end
    tuner_new = MCMCFlowTuner(AdaptiveFlows.Adam(eta), tuner.n_batches,tuner.n_epochs) # might want to update the training parameters 

    return tuner_new, flow_new
end

tuning_postinit!(tuner::MCMCFlowTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::MCMCFlowTuner, chain::MCMCIterator, max_nsteps::Integer) = nothing

tuning_update!(tuner::MCMCFlowTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_finalize!(tuner::MCMCFlowTuner, chain::MCMCIterator) = nothing

tuning_callback(::MCMCFlowTuning) = nop_func

#######Delete this section later; just for debug purposes

# function bat_sample_impl_ensemble(
#     target::BATMeasure,
#     algorithm::TransformedMCMC,
#     context::BATContext
# )  

#     density, pre_trafo = transform_and_unshape(algorithm.pretransform, target, context)
#     vs = varshape(target)

#     init = mcmc_init!(
#         algorithm,
#         density,
#         algorithm.nchains,
#         apply_trafo_to_init(pre_trafo, algorithm.init),
#         #algorithm.tuning_alg,
#         algorithm.transform_tuning,
#         algorithm.nonzero_weights,
#         algorithm.store_burnin ? algorithm.callback : nop_func,
#         context
#     )

#     @unpack ensembles, tuners, temperers = init

#     burnin_outputs_coll = if algorithm.store_burnin
#         DensitySampleVector(first(ensembles))
#     else
#         nothing
#     end

#     # burnin and tuning  # @TODO: Hier wird noch kein ensemble BurnIn gemacht !!!!!!!!!!!!!!!
#    # mcmc_burnin!(
#    #     burnin_outputs_coll,
#    #     ensembles,
#    #     tuners,
#    #     temperers,
#    #     algorithm.burnin,
#    #     algorithm.convergence,
#    #     algorithm.strict,
#    #     algorithm.nonzero_weights,
#    #     algorithm.store_burnin ? algorithm.callback : nop_func
#    # )


#     # sampling
#     run_sampling = _run_sample_impl_ensemble(
#         density,
#         algorithm,
#         ensembles,
#         tuners,
#         temperers
#     )

#     samples_trafo, generator, flow = run_sampling.result_trafo, run_sampling.generator, run_sampling.flow

#     # prepend burnin samples to output
#     if algorithm.store_burnin
#         burnin_samples_trafo = burnin_outputs_coll
#         prepend!(samples_trafo, burnin_samples_trafo)
#     end

#     global g_state_post_algorithm = (pre_trafo, samples_trafo, vs, density, algorithm, ensembles, tuners, temperers, target, context)

#     samples_notrafo = inverse(pre_trafo).(samples_trafo)
#     #samples_notrafo = vs.(inverse(pre_trafo).(samples_trafo))

#     (result = samples_notrafo, result_trafo = samples_trafo, trafo = pre_trafo, generator = MCMCSampleGenerator(ensembles), flow = flow)
# end

# function bat_sample_impl(
#     target::BATMeasure,
#     algorithm::TransformedMCMC,
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
    

#     (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = MCMCSampleGenerator(chains))
# end