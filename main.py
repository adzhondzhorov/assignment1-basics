
import datetime

from cs336_basics.training.train_loop_owt import print_log, train


if __name__ == "__main__":
    ITERS = 13000

    print_log("======================================================================================")
    print_log("MAKE SURE TO RUN ONCE TO CACHE THE TOKENIZED TRAIN AND VAL SETS BEFORE BENCHMARK RUN.")
    print_log("======================================================================================")
    print_log(str(datetime.datetime.now()))
    for optimizer_params in [
        dict(
            alpha=None,
            lr_schedule=dict(
                alpha_max=3e-3, alpha_min=5e-6, t_warmup=100, t_final=ITERS
            ),
            lambda_=0.1,
            beta1=0.9,
            beta2=0.975,
            epsilon=1e-8,
        ),
    ]:
        print_log(optimizer_params)
        llm = train(optimizer_params, iterations=ITERS, batch_size=32)
