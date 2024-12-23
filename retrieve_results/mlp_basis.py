import wandb
import numpy as np
import time
import argparse


def get_runs():
    """
    Collect Runs from WandB.

    Output:
        - runs from WandB.
    """
    api = wandb.Api()

    # Define the project
    project_name = "vector_final3"
    # project_name = "ThesisFinal2Grav+coll"
    project = f"lveefkind/{project_name}"

    runs = api.runs(project)
    print(f"There are {len(runs)} runs in {project_name}.")
    return runs


def get_grouped_filtered_runs(runs, filters, groups):
    """
    Group and filter runs.

    Input:
        - runs: runs from WandB.
        - filters: dictionary with filters.
        - groups: variables to group by.

    Output:
        - Filtered and grouped runs.
    """
    filtered_runs = []
    grouped_runs = {}
    for run in runs:
        config = run.config
        # If it satisfies all filters, save it
        if (
            all([config.get(filter) == filters[filter] for filter in filters])
            and ("lessSimulations??" not in run.tags)
            and (run.state != "crashed")
            and (run.state != "failed")
        ):
            filtered_runs += [run]
            config = run.config

            # Place run in correct group
            key = tuple([config.get(group) for group in groups])
            if key in grouped_runs:
                grouped_runs[key].append(run)
            else:
                grouped_runs[key] = [run]
    print(f"{len(filtered_runs)} runs satisfy the filters {filters}.")
    return filtered_runs, grouped_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--new_collect_results",
        action="store_true",
        help="Force the model to focus on identity",
    )
    args = parser.parse_args()

    # Define train directory.
    # train_dir = "data_t(5,20)_r(0,0)_combi_pNone_gNone"
    # train_dir = "data_t(0,0)_r(5,20)_combi_pNone_gNone"
    train_dir = "data_t(5,20)_r(5,20)_combi_pNone_gNone"
    # train_dir = "data_t(0,0)_r(0,0)_combi_pNone_gTrue"
    # train_dir = "data_t(5,20)_r(5,20)_combi_pTrue_gTrue"
    # train_dir = "data_t(0,0)_r(5,20)_combi_pNone_gTrue"
    # train_dir = "data_tennis_pNone_gNone_tennisEffect"

    # Define filters
    filters = {
        "angle": True,
        "norm": True,
        "one_eq": False,
        "channel_splits": False,
        "gated": False,
    }

    # Collect results from WandB
    if args.new_collect_results:
        start_time = time.time()
        # Collect runs
        runs = get_runs()
        group_by = ["equivariance"]
        # Filter and group runs
        filtered_runs, grouped_runs = get_grouped_filtered_runs(
            runs,
            filters,
            group_by,
        )

        # Calculate average minimum of each group
        # scores = [
        #     score
        #     for score in filtered_runs[0].history().keys()
        #     if ("loss" in score and "Test" in score)
        # ]
        scores = ["Test total loss"]
        for key, runs in (grouped_runs).items():
            for score in scores:
                values = []
                for run in runs:
                    values.append(run.history().get(score)[100])
                # print(key, score, f"{np.mean(values):.3f}", f"{np.std(values):.3f}")
                print(
                    key, score, f"${np.mean(values):.3f} \ (\pm {np.std(values):.3f})$"
                )
        exit()

        average_data = average_runs(grouped_runs, train_dir[5:])

        # Filter extra if necessary.
        specific_df = get_specific_values(
            filters,
            average_data,
        )
        print(f"It took {time.time() - start_time} seconds to get all results!")
    # Calculate from already collected results
    else:
        # Filter runs.
        print("Using already collected runs !(These may be old)!")
        specific_df = get_specific_values(filters)

    # drop the columns where all values are NaN
    remove_empty_df = specific_df.replace("", np.nan)
    remove_empty_df = remove_empty_df.dropna(how="all", axis=1)
    print(remove_empty_df)
    # Store as CSV file
    remove_empty_df.to_csv("try_out.csv", index=False)
