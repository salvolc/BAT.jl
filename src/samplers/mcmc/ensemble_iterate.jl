mutable struct TransformedMCMCEnsembleIterator{
    PR<:RNGPartition,
    D<:BATMeasure,
    F<:Function,
    Q<:MCMCProposal,
    SV,#<:Vector{DensitySampleVector},
    S<:DensitySampleVector,
    CTX<:BATContext,
} <: MCMCIterator
    rngpart_cycle::PR
    μ::D
    f::F
    proposal::Q
    states_x::SV
    state_z::S
    stepno::Int
    n_walker::Int
    tau::Float64
    use_mala::Bool
    n_accepted::Vector{Int}
    info::MCMCChainStateInfo
    context::CTX
end

mcmc_target(ensemble::TransformedMCMCEnsembleIterator) = ensemble.μ

getmeasure(ensemble::TransformedMCMCEnsembleIterator) = ensemble.μ

get_context(ensemble::TransformedMCMCEnsembleIterator) = ensemble.context

mcmc_info(ensemble::TransformedMCMCEnsembleIterator) = ensemble.info 

nsteps(ensemble::TransformedMCMCEnsembleIterator) = ensemble.stepno

nsamples(ensemble::TransformedMCMCEnsembleIterator) = length(ensemble.states_x) * length(ensemble.states_x[1]) # @TODO

current_ensemble(ensemble::TransformedMCMCEnsembleIterator) = last(ensemble.states_x)

sample_type(ensemble::TransformedMCMCEnsembleIterator) = eltype(ensemble.state_z)

samples_available(ensemble::TransformedMCMCEnsembleIterator) = size(ensemble.states_x,1) > 0

isvalidchain(ensemble::TransformedMCMCEnsembleIterator) = min(current_sample(ensemble).logd) > -Inf

isviablechain(ensemble::TransformedMCMCEnsembleIterator) = nsamples(ensemble) >= 2

eff_acceptance_ratio(ensemble::TransformedMCMCEnsembleIterator) = nsamples(ensemble) / ensemble.stepno


#ctor
function TransformedMCMCEnsembleIterator(
    algorithm::TransformedMCMC,
    target,
    id::Integer,
    v_init::AbstractVector{<:Real},
    context::BATContext
) 
    TransformedMCMCEnsembleIterator(algorithm, target, Int32(id), v_init, context)
end

export TransformedMCMCEnsembleIterator


function TransformedMCMCEnsembleIterator(
    algorithm::TransformedMCMC,
    target,
    id::Int32,
    v_init::AbstractVector{},
    context::BATContext,
)
    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))

    μ = target
    proposal = algorithm.proposal
    stepno = 1
    n_walker= algorithm.init.nwalker
    tau = algorithm.init.tau
    use_mala = algorithm.init.use_mala
    cycle = 1
    n_accepted = zeros(Int64,length(v_init))

    adaptive_transform_spec = algorithm.adaptive_transform
    f = init_adaptive_transform(adaptive_transform_spec, μ, context)

    logd_x = BAT.checked_logdensityof(μ).(v_init)
    states = [DensitySampleVector((v_init, logd_x, ones(length(logd_x)), fill(TransformedMCMCTransformedSampleID(id, 1, 0),length(logd_x)), fill(nothing,length(logd_x))))]

    global g_state = (f,states)
    state_z = states[end]#f(states[end])
    
    iter = TransformedMCMCEnsembleIterator(
        rngpart_cycle,
        target,
        f,
        proposal,
        states,
        state_z,
        stepno,
        n_walker,
        tau,
        use_mala,
        n_accepted,
        MCMCChainStateInfo(id, cycle, false, false),
        context
    )
end

# This function is used for adaptive Transformations which arent flow, not sure if it is needed for Ensembles
function propose_mcmc(
    iter::TransformedMCMCEnsembleIterator{<:Any, <:Any, <:Any, <:MCMCProposal}
)
    @unpack μ, f, proposal, states_x, state_z, stepno, context = iter
    rng = get_rng(context)
    states_x = last(states_x)
    x, logd_x = states_x.v, states_x.logd
    z, logd_z = state_z.v, state_z.logd
    f_inv = inverse(f)
    #n, m = size(z)

    z_proposed = similar(z)
    for i in 1:length(z)
        z_proposed[i] = z[i] + rand(rng,MvNormal(zeros(length(z[i])),ones(length(z[i])))) #TODO: check if proposal is symmetric? otherwise need additional factor?
    end  
    #z_proposed = z + rand(rng, proposal.proposal_dist, (n,m)) #TODO: check if proposal is symmetric? otherwise need additional factor?       
    x_proposed =z_proposed#, ladj = with_logabsdet_jacobian(f_inv, z_proposed)
    logd_x_proposed = BAT.checked_logdensityof(μ).(x_proposed)
    logd_z_proposed = logd_x_proposed #+ vec(ladj)

    #@assert logd_z_proposed ≈ logdensityof(MeasureBase.pullback(f_inv, μ)).(z_proposed) #TODO: remove

    p_accept = clamp.(exp.(logd_z_proposed-logd_z), 0, 1)

    state_z_proposed = _rebuild_density_sample_vector(state_z, z_proposed, logd_z_proposed)
    state_x_proposed = _rebuild_density_sample_vector(states_x, x_proposed, logd_x_proposed)

    return state_x_proposed, state_z_proposed, p_accept
end

# Used to sample directly from flow
function propose_random_normal(dim::Int, n::Int,rng)
    return rand(rng,MvNormal(zeros(dim),ones(dim)),n)
end

function propose_mcmc(z, dim::Int, rng)
    return z + rand(rng, MvNormal(zeros(dim),ones(dim)))
end

#function propose_mala(z, dim::Int, tau::Float64, gradient::AbstractVector, rng)
#    tau=0.2
#    return z + sqrt(2*tau) * rand(rng, MvNormal(zeros(dim),ones(dim)))+ tau .* gradient
#end
function propose_mala(z, dim, tau, gradient, rng)
    proposal = z + 0.5 * tau^2 * gradient + sqrt(tau) * rand(rng, MvNormal(zeros(dim),ones(dim)))
    #proposal_covariance = tau^2 * I(dim)

    #proposal = rand(rng, MvNormal(proposal_mean, proposal_covariance))
    
    return proposal
end


function propose_state(
    iter::TransformedMCMCEnsembleIterator{<:Any, <:Any, <:Any, <:MCMCProposal}, flow_rate::Float64= 0.00 # TODO Hardcode->Parameter
)
    @unpack μ, f, proposal, states_x, state_z, stepno, context = iter
    rng = get_rng(context)
    AD_sel = context.ad
    f_inv = inverse(f)
    z = state_z.v
    logd_x = last(states_x).logd

    dim=length(z[1])
    z_proposed = similar(z)

    flow_mask = (rand(rng,length(z)) .< flow_rate)
    z_proposed[flow_mask] = nestedview(propose_random_normal(dim,sum(flow_mask),rng))
    
    ν = Transformed(convert(BATMeasure,μ), f, TDLADJCorr())
    log_ν = BAT.checked_logdensityof(ν)
    
    if iter.use_mala == true
        ∇log_ν = gradient_func(log_ν, AD_sel)
        grads = ∇log_ν.(z)

        i = .!flow_mask
        z_proposed[i] = propose_mala.(z[i], dim, iter.tau, grads[i],rng)
    else
        i = .!flow_mask
        z_proposed[i] = propose_mcmc.(z[i], dim, rng)
    end

    x_proposed, ladj = with_logabsdet_jacobian(f_inv, z_proposed)
    logd_x_proposed = BAT.checked_logdensityof(μ).(x_proposed)    
    logd_z_proposed = vec(logd_x_proposed + ladj')

    for i in 1:length(x_proposed)
        if any(isnan.(x_proposed[i]))
            x_proposed[i] = last(states_x).v[i]
            error("Fucking NaN!")
        end
    end

    p_accept = clamp.(exp.(logd_x_proposed-logd_x), 0, 1)

    state_x_proposed = _rebuild_density_sample_vector(last(states_x), x_proposed, logd_x_proposed)
    state_z_proposed = _rebuild_density_sample_vector(state_z, z_proposed, logd_z_proposed)

    return state_x_proposed, state_z_proposed, p_accept
end

function adaptive_tau_update!(iter, actuel_rate, target_accept_rate)
    gamma = 0.01
    factor = gamma * (actuel_rate - target_accept_rate) / target_accept_rate
    iter.tau *= exp(factor)
    return iter.tau
end
    
function transformed_mcmc_step!!(
    iter::TransformedMCMCEnsembleIterator,
    tuner::MCMCTransformTunerState,
    tempering::TemperingState,
)
    @unpack  μ, f, proposal, states_x, state_z, stepno, n_accepted, context = iter
    rng = get_rng(context)
    state_x = last(states_x)

    global g_state = (iter)
    if f isa Mul
        state_x_proposed, state_z_proposed, p_accept = propose_mcmc(iter)
    else
        state_x_proposed, state_z_proposed, p_accept = propose_state(iter)
    end

    z_proposed, logd_z_proposed = state_z_proposed.v, state_z_proposed.logd
    x_proposed, logd_x_proposed = state_x_proposed.v, state_x_proposed.logd
    
    accepted = rand(rng, length(p_accept)) .<= p_accept
    rate = sum(accepted)/length(p_accept)
    adaptive_tau_update!(iter,rate,0.57)

    x_new, logd_x_new = copy(state_x.v), copy(state_x.logd)
    z_new, logd_z_new = copy(state_z.v), copy(state_z.logd)

    x_new[accepted], z_new[accepted], logd_x_new[accepted], logd_z_new[accepted] = x_proposed[accepted], z_proposed[accepted], logd_x_proposed[accepted], logd_z_proposed[accepted]

    state_x_new = DensitySampleVector((x_new, logd_x_new,ones(length(x_new)), fill(TransformedMCMCTransformedSampleID(iter.info.id, iter.info.cycle, iter.stepno), length(x_new)), fill(nothing,length(x_new))))

    push!(states_x, state_x_new) 

    state_z_new = _rebuild_density_sample_vector(state_z, z_new, logd_z_new)

    #tuner_new=tuner
    if (tuner isa MCMCFlowTuner)
        tuner_new, f_new = tune_mcmc_transform!!(tuner, f, state_x_new.v, μ, context)
        f=f_new.result
        loss = f_new.loss_hist[2][1]
        open("/nfs/homes/slacagnina/Documents/Transformed_BAT-RenameTransformed/src/data.txt", "a") do file
            write(file, "$loss")
        end
        #println(tuner.optimizer.eta)
        #println(tuner_new.optimizer.eta)
    else #(tuner isa NoMCMCTransformTuning)
        global g_state = (tuner, f, p_accept, z_proposed, state_z.v, stepno, context)
        tuner_new, f = tune_mcmc_transform!!(tuner, f, p_accept, z_proposed, state_z.v, stepno, context)
        #mcmc_tune_post_cycle!!(tuner)
        #tuner_new = tuner
        #f = f
        #tuner_new, f = tune_mcmc_transform!!(tuner, f, p_accept, z_proposed, state_z.v, stepno, context)
        # should this take the old z state?
    end
    tempering_new, μ_new = temper_mcmc_target!!(tempering, μ, stepno)
    #tuner = tuner_new

    f_new = f

    iter.μ, iter.f, iter.states_x, iter.state_z = μ_new, f_new, states_x, state_z_new
    iter.n_accepted += Int.(accepted)
    iter.stepno += 1
    @assert iter.context === context

    return (iter, tuner_new, tempering_new)
end


function transformed_mcmc_iterate!( 
    ensembles::AbstractVector{<:TransformedMCMCEnsembleIterator},
    tuners::AbstractVector{<:MCMCTransformTunerState},
    temperers::AbstractVector{<:TemperingState};
    kwargs...
)
    global a3 = tuners
    if isempty(ensembles)
        @debug "No MCMC ensemble(s) to iterate over."
        return ensembles
    else
        @debug "Starting iteration over $(length(ensembles)) MCMC ensemble(s)"
    end
    
    for i in 1:length(ensembles)
        ensembles[i], tuners[i],temperers[i] = transformed_mcmc_step!!(ensembles[i], tuners[i],temperers[i])
    end

    return nothing
end


function reset_rng_counters!(ensemble::TransformedMCMCEnsembleIterator)
    rng = get_rng(get_context(ensemble))
    set_rng!(rng, ensemble.rngpart_cycle, ensemble.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, ensemble.stepno)
    nothing
end


function next_cycle!(
    ensemble::TransformedMCMCEnsembleIterator,

)
    ensemble.info = MCMCChainStateInfo(ensemble.info, cycle = ensemble.info.cycle + 1)
    ensemble.stepno = 0

    reset_rng_counters!(ensemble)

    ensemble.states_x[1] = last(ensemble.states_x)
    resize!(ensemble.states_x, 1)

    ensemble.states_x[1].weight[1] = 1
    ensemble.states_x[1].info[1] = TransformedMCMCTransformedSampleID(ensemble.info.id, ensemble.info.cycle, ensemble.stepno)
    
    ensemble
end

function _rebuild_density_sample_vector(s::DensitySampleVector, x, logd, weight=ones(length(x)))
    @unpack info, aux = s
    DensitySampleVector((x, logd, weight, info, aux))
end