from .corrpops_logger import corrpops_logger


_ = corrpops_logger()


# corrpops_logger().info(
#     "\n\033[93m"
#     "corrpops logs and writes progress to stdout.\n"
#     "> to set optimizer logging to DEBUG, set CorrPopsOptimizer(verbose=False) "
#     'or CorrPopsEstimator(optimizer_kwargs={"verbose": False})\n'
#     "  \x1B[3m(this is recommended when running CorrPopsJackknifeEstimator.fit, for example).\x1B[0m"
#     "\n\033[93m"
#     "> to disable all logging from corrpops, run `corrpops.corrpops_logger().setLevel(30)`"
#     "\033[0m"
# )
