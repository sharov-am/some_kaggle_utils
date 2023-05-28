def timed_optuna(study, objective, timeout=5, iters=100, es=10):

    study.optimize(objective, n_trials=iters, timeout=timeout*60,
                   callbacks=[partial(early_stopping, rounds=es)])

    print(f"Num trials:  {len(study.trials)}")
    print(f"Best score:  {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    fig = plot_optimization_history(study)
    fig.show(config={"staticPlot": True})