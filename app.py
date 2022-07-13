from itertools import combinations
import image_generator as IG
import nn_model as NN
from calculate_metrics import calc_combinations, determine_best_combination_and_evaluate


def main():

    ig = IG.ImageGenerator(data_name="OIH_adjusted.txt")
    ig.generate_images()

    nn = NN.NNModel(df_closing_prices=ig.df_closing_prices, NN_number=9, patience=20, verbose=0)
    nn.create_train_and_evaluate_model()

    initializers_labels = ["GN", "GU", "HN", "HU", "LU", "OR", "RN", "RU", "TN", "VS"]
    train_starts = [
        "2001-04-02",
        "2001-07-02",
        "2001-10-01",
        "2002-01-02",
        "2002-04-01",
        "2002-07-01",
        "2002-10-01",
        "2003-01-02",
        "2003-04-01",
    ]
    combinations_list = sum(
        [
            list(map(list, combinations(initializers_labels, i)))
            for i in range(len(initializers_labels) + 1)
        ],
        [],
    )

    nn_type = "MAIN"
    predictions_dir_name_nn_type = [("PREDICTIONS", nn_type)]
    combinations_results_dir_name = predictions_dir_name_nn_type[0][0] + "_MAX_PREDICTIONS/"

    calc_combinations(
        combinations_list,
        train_starts,
        initializers_labels,
        predictions_dir_name_nn_type,
        combinations_results_dir_name,
        "ALL_AT_ONCE",
    )
    calc_combinations(
        combinations_list,
        train_starts,
        initializers_labels,
        predictions_dir_name_nn_type,
        combinations_results_dir_name,
        "ONE_BY_ONE",
    )
    determine_best_combination_and_evaluate(
        combinations_list, train_starts, nn_type, combinations_results_dir_name
    )


if __name__ == "__main__":

    main()
