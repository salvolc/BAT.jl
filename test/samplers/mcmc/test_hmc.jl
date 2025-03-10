# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes, ArraysOfArrays, DensityInterface
using IntervalSets
import ForwardDiff, Zygote

import AdvancedHMC

@testset "HamiltonianMC" begin
    context = BATContext(ad = ForwardDiff)
    objective = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_target = @inferred(batmeasure(objective))
    @test shaped_target isa BAT.BATDistMeasure
    target = unshaped(shaped_target)
    @test target isa BAT.BATDistMeasure

    proposal = HamiltonianMC()
    transform_tuning = StanLikeTuning()
    nchains = 4
    samplingalg = TransformedMCMC(proposal = proposal, transform_tuning = transform_tuning, nchains = nchains)

    @testset "MCMC iteration" begin
        v_init = bat_initval(target, InitFromTarget(), context).result
        # Note: No @inferred, since MCMCChainState is not type stable (yet) with HamiltonianMC
        @test BAT.MCMCChainState(samplingalg, target, 1, unshaped(v_init, varshape(target)), deepcopy(context)) isa BAT.HMCState
        mcmc_state = BAT.MCMCState(samplingalg, target, 1, unshaped(v_init, varshape(target)), deepcopy(context))
        nsteps = 10^4
        BAT.mcmc_tuning_init!!(mcmc_state, 0)
        BAT.mcmc_tuning_reinit!!(mcmc_state, div(nsteps, 10))

        samples = DensitySampleVector(mcmc_state)
        mcmc_state = BAT.mcmc_iterate!!(samples, mcmc_state; max_nsteps = nsteps, nonzero_weights = false)
        @test mcmc_state.chain_state.stepno == nsteps
        @test minimum(samples.weight) == 0
        @test isapprox(length(samples), nsteps, atol = 20)
        @test length(samples) == sum(samples.weight)
        @test BAT.test_dist_samples(unshaped(objective), samples)

        samples = DensitySampleVector(mcmc_state)
        mcmc_state = BAT.mcmc_iterate!!(samples, mcmc_state; max_nsteps = 10^3, nonzero_weights = true)
        @test minimum(samples.weight) == 1
    end

    @testset "MCMC tuning and burn-in" begin
        max_nsteps = 10^5
        transform_tuning = BAT.StanLikeTuning()
        pretransform = DoNotTransform()
        init_alg = bat_default(TransformedMCMC, Val(:init), proposal, pretransform, nchains, max_nsteps)
        burnin_alg = bat_default(TransformedMCMC, Val(:burnin), proposal, pretransform, nchains, max_nsteps)
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing
    
        samplingalg = TransformedMCMC(proposal = proposal,
            transform_tuning = transform_tuning, 
            pretransform = pretransform, 
            init = init_alg, 
            burnin = burnin_alg, 
            convergence = convergence_test, 
            strict = strict, 
            nonzero_weights = nonzero_weights
        )
    
        # Note: No @inferred, not type stable (yet) with HamiltonianMC
        init_result = BAT.mcmc_init!(
            samplingalg,
            target,
            init_alg,
            callback,
            context
        )
    
        (mcmc_states, outputs) = init_result
        @test mcmc_states isa AbstractVector{<:BAT.MCMCState}
        @test outputs isa AbstractVector{<:DensitySampleVector}
    
        mcmc_states = BAT.mcmc_burnin!(
            outputs,
            mcmc_states,
            samplingalg,
            callback
        )
    
        BAT.next_cycle!.(mcmc_states)
    
        mcmc_states = BAT.mcmc_iterate!!(
            outputs,
            mcmc_states;
            max_nsteps = div(max_nsteps, length(mcmc_states)),
            nonzero_weights = nonzero_weights
        )
    
        samples = DensitySampleVector(first(mcmc_states))
        append!.(Ref(samples), outputs)
        
        @test length(samples) == sum(samples.weight)
        @test BAT.test_dist_samples(unshaped(objective), samples)
    end
    
    @testset "bat_sample" begin
        samples = bat_sample(
            shaped_target,
            TransformedMCMC(
                proposal = proposal,
                transform_tuning = StanLikeTuning(),
                pretransform = DoNotTransform(),
                nsteps = 10^4,
                store_burnin = true
            ),
            context
        ).result

        # ToDo: First HMC sample currently had chaincycle set to 0, should be fixed.
        # @test first(samples).info.chaincycle == 1
        @test samples[2].info.chaincycle == 1

        smplres = BAT.sample_and_verify(
            shaped_target,
            TransformedMCMC(
                proposal = proposal,
                transform_tuning = StanLikeTuning(),
                pretransform = DoNotTransform(),
                nsteps = 10^4,
                store_burnin = false
            ),
            objective,
            context
        )
        samples = smplres.result
        @test first(samples).info.chaincycle >= 2
        @test samples.v isa ShapedAsNTArray
        @test smplres.verified
    end

    @testset "MCMC sampling in transformed space" begin
        prior = BAT.example_posterior().prior
        likelihood = logfuncdensity(v -> 0)
        inner_posterior = PosteriorMeasure(likelihood, prior)
        # Test with nested posteriors:
        posterior = PosteriorMeasure(likelihood, inner_posterior)
        @test BAT.sample_and_verify(posterior, TransformedMCMC(proposal = HamiltonianMC(), transform_tuning = StanLikeTuning(), pretransform = PriorToNormal()), prior.dist, context).verified
    end

    @testset "HMC autodiff" begin
        posterior = BAT.example_posterior()

        for admodule in [ForwardDiff, Zygote]
            @testset "$admodule" begin
                context = BATContext(ad = admodule)

                hmc_samplingalg = TransformedMCMC(
                    proposal = HamiltonianMC(),
                    transform_tuning = StanLikeTuning(),
                    nchains = 2,
                    nsteps = 100,
                    init = MCMCChainPoolInit(init_tries_per_chain = 2..2, nsteps_init = 5),
                    burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 100, max_ncycles = 1),
                    strict = false
                )
                
                @test bat_sample(posterior, hmc_samplingalg, context).result isa DensitySampleVector
            end
        end
    end
end
