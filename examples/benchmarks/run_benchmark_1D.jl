function run_1D_benchmark(;algorithm=RandomWalk(), nsteps=10^5, nchains=8)
    for i in 1:length(testfunctions_1D)
        sample_stats_all = run1D(
            collect(keys(testfunctions_1D))[i], #There might be a nicer way but I need the name to save the plots
            testfunctions_1D,
            sample_stats[i],
            run_stats[i],
            algorithm,
            nsteps,
            nchains
        )
    end
    make_1D_results(testfunctions_1D,sample_stats)
    save_stats_1D(collect(keys(testfunctions_1D)),run_stats,run_stats_names)
end
